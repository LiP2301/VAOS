import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from controllers.basic_controller import BasicMAC
import torch as th
from torch.optim import RMSprop
import numpy as np


class VAOSLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        args.trans = 2
        self.idx = 0
        self.idxm = 0
        self.omega = 10
        self.qtot = 1000000
        self.qi = 100000
        self.tindrnn = th.zeros(1, self.args.k)
        self.tindmix = th.zeros(1, self.args.km)
        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)
            if args.km > 1:
                self._target_mixer = [copy.deepcopy(self.mixer) for _ in range(args.km)]
            else:
                self._target_mixer = copy.deepcopy(self.mixer)
            for i in range(args.km):
                self._target_mixer[i].load_state_dict(self.mixer.state_dict())
                # self._target_mixer[i].cuda()

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)
        if args.k > 1:
            self._target_mac = [copy.deepcopy(mac) for _ in range(args.k)]
        else:
            self._target_mac = copy.deepcopy(mac)
        if args.k > 1:
            for i in range(args.k):
                self._target_mac[i].load_state(self.mac)
                # self._target_mac[i].cuda()
        else:
            self._target_mac.load_state(self.mac)
            # self._target_mac.agent = self._target_mac.agent.cuda

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        ave = 0
        target_mac_out = []
        # Calculate the Q-Values necessary for the target
        if ave:
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)
        else:
            qind = th.zeros(1, self.args.k)
            for i in range(self.args.k):
                qind[0, i] = (i + 1) * self.qi / self.args.k
            for j in range(self.args.k):
                if t_env < qind[0, j]:
                    qin_t = j + 1
                    break
                else:
                    qin_t = self.args.k
            self.target_mac.init_hidden(batch.batch_size)
            for y in range(self.args.k):
                self._target_mac[y].init_hidden(batch.batch_size)
                # self._target_mac[y].hidden_states = self._target_mac[y].hidden_states.cuda()
            for t in range(batch.max_seq_length):
                # target_agent_outs = self._target_mac[i].forward(batch, t=t)
                q_target = th.zeros(agent_outs.shape[0], agent_outs.shape[1], agent_outs.shape[2])
                # q_target = q_target.cuda()
                _, rf = th.sort(self.tindrnn, descending=True)
                for i in range(qin_t):
                    dd = rf[0, i]
                    target_agent_outs = self._target_mac[rf[0, i]].forward(batch, t=t)
                    q_target += target_agent_outs
                q_target /= qin_t
                target_mac_out.append(q_target)

        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time
        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            if t_env % (self.args.trans*25) == 0:
                dd = th.exp(self.omega * (target_mac_out))
                df = th.sum(dd, -1, keepdim=True) / 5
                mellow = th.log(df) / self.omega
                target_max_qvals = self.target_mixer(mellow, batch["state"][:, 1:])
            else:
                target_max_qvals = target_mac_out.max(dim=3)[0]
                mix_qind = th.zeros(1, self.args.km)
                for i in range(self.args.km):
                    mix_qind[0, i] = (i + 1) * self.qtot / self.args.km
                for j in range(self.args.km):
                    if t_env < mix_qind[0, j]:
                        mix_qin_t = j + 1
                        break
                    else:
                        mix_qin_t = self.args.km
                mix_q_target = th.zeros(target_max_qvals.shape[0], target_max_qvals.shape[1], 1)
                # mix_q_target = mix_q_target.cuda()
                _, rfm = th.sort(self.tindmix, descending=True)
                for i in range(mix_qin_t):
                    tar_max_qvals = self._target_mixer[rfm[0, i]](target_max_qvals, batch["state"][:, 1:])  #
                    mix_q_target += tar_max_qvals
                mix_q_target /= mix_qin_t
                target_max_qvals = mix_q_target

        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals
        # Td-error
        td_error = (chosen_action_qvals - targets.detach())
        mask = mask.expand_as(td_error)
        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask
        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        if self.args.vaos:
            future_episode_return = batch["future_discounted_return"][:, :-1]
            q_return_diff = (chosen_action_qvals - future_episode_return.detach())
            vaos_l2 = ((q_return_diff * mask) ** 2).sum() / mask.sum()
            loss += self.args.vaos_lambda * vaos_l2

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

            if self.args.k > 1:
                self._target_mac[self.idx % self.args.k].load_state(self.mac)
            else:
                self._target_mac.load_state(self.mac)
            self.tindrnn[0, self.idx % self.args.k] = self.idx + 1
            self.idx += 1

            if self.args.km > 1:
                self._target_mixer[self.idxm % self.args.km].load_state_dict(self.mixer.state_dict())
            else:
                self._target_mixer.load_state_dict(self.mixer.state_dict())
            self.tindmix[0, self.idxm % self.args.km] = self.idxm + 1
            self.idxm += 1

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean",
                                 (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                                 t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))


