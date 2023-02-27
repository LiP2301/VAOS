import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from controllers.basic_controller import BasicMAC
import torch as th
from torch.optim import RMSprop
import numpy as np


class QLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        args.k = 5
        args.km = 10
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

    def softmax_weighting(self, q_vals):
        assert q_vals.shape[-1] != 1

        max_q_vals = th.max(q_vals, -1, keepdim=True)[0]
        norm_q_vals = q_vals - max_q_vals
        e_beta_normQ = th.exp(self.args.res_beta * norm_q_vals)

        numerators = e_beta_normQ
        denominators = th.sum(e_beta_normQ, -1, keepdim=True)

        softmax_weightings = numerators / denominators

        return softmax_weightings

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
        targdd = []
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
        # Max over target Q-Values
        # if self.args.double_q:
        #     # Get actions that maximise live Q (for double q-learning)
        #     mac_out_detach = mac_out.clone().detach()
        #     mac_out_detach[avail_actions == 0] = -9999999
        #     cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
        #     target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        # else:
        #     target_max_qvals = target_mac_out.max(dim=3)[0]
        # target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            # mix_qind = th.zeros(1, self.args.km)
            # for i in range(self.args.km):
            #     mix_qind[0, i] = (i + 1) * 1000000 / self.args.km
            # for j in range(self.args.km):
            #     if t_env < mix_qind[0, j]:
            #         mix_qin_t = j + 1
            #         break
            #     else:
            #         mix_qin_t = self.args.km
            # 0 t_env % (self.args.trans*25) == 0  t_env > 1000000
            if t_env % (self.args.trans*25) == 0:
                # c = th.max(target_mac_out,3).values.unsqueeze(3)
                # dd = th.exp(self.omega * (target_mac_out - c))
                # # rf = dd.reshape(-1, dd.shape[2],dd.shape[3])
                # df = th.sum(dd, -1, keepdim=True)/5
                # mellow = c + th.log(df) / self.omega
                # ans = th.sum(th.exp((target_mac_out - mellow) * self.beta) * (target_mac_out - mellow), -1, keepdim=True)
                # target_max_qvals = self.target_mixer(ans, batch["state"][:, 1:])
                dd = th.exp(self.omega * (target_mac_out))
                df = th.sum(dd, -1, keepdim=True) / 5
                mellow = th.log(df) / self.omega
                target_max_qvals = self.target_mixer(mellow, batch["state"][:, 1:])
                # mac_out_detach = mac_out.clone().detach()
                # mac_out_detach[avail_actions == 0] = -9999999
                # cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
                # target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
                # all_counterfactual_actions_qvals = []
                # all_counterfactual_actions_target_qvals = []
                # for agent_idx in range(cur_max_actions.shape[2]):
                #     base_actions = copy.deepcopy(cur_max_actions)
                #     # total_batch_size, num_agents
                #     base_actions = base_actions.squeeze(-1).reshape(-1, cur_max_actions.shape[2])
                #     # num_actions, 1
                #     all_actions_for_an_agent = th.tensor([action_idx for action_idx in range(self.args.n_actions)]).unsqueeze(0)
                #     # num_actions, total_batch_size: [[0, ..., 0], [1, ..., 1], ..., [4, ..., 4]]
                #     all_actions_for_an_agent = all_actions_for_an_agent.repeat(base_actions.shape[0], 1).transpose(1, 0)
                #     # formate to a column vector: total_batch_size x num_actions: [0, ..., 0, ...., 4, ..., 4]
                #     all_actions_for_an_agent = all_actions_for_an_agent.reshape(-1, 1).squeeze()
                #     # total_batch_size x num_agents, num_actions (repeat the actions for num_actions times)
                #     counterfactual_actions = base_actions.repeat(self.args.n_actions, 1).reshape(-1, base_actions.shape[1])
                #     counterfactual_actions[:, agent_idx] = all_actions_for_an_agent
                #     counterfactual_actions_qvals, counterfactual_actions_target_qvals = [], []
                #     for action_idx in range(self.args.n_actions):
                #         curr_counterfactual_actions = counterfactual_actions[action_idx * base_actions.shape[0] : (action_idx + 1) * base_actions.shape[0]]
                #         curr_counterfactual_actions = curr_counterfactual_actions.reshape(cur_max_actions.shape[0], cur_max_actions.shape[1], cur_max_actions.shape[2], -1)
                #         # batch_size, episode_len, num_agents
                #         curr_counterfactual_actions_qvals = th.gather(mac_out_detach[:, 1:], dim=3, index=curr_counterfactual_actions).squeeze(3)  # Remove the last dim
                #         curr_counterfactual_actions_target_qvals = th.gather(target_mac_out, dim=3, index=curr_counterfactual_actions).squeeze(3)  # Remove the last dim
                #         # batch_size, episode_len, 1
                #         curr_counterfactual_actions_qvals = self.mixer(curr_counterfactual_actions_qvals, batch["state"][:, 1:])
                #         curr_counterfactual_actions_qvals = curr_counterfactual_actions_qvals.reshape(curr_counterfactual_actions_qvals.shape[0] * curr_counterfactual_actions_qvals.shape[1], 1)
                #
                #         curr_counterfactual_actions_target_qvals = self.target_mixer(curr_counterfactual_actions_target_qvals, batch["state"][:, 1:])
                #         # mix_q_target = th.zeros(target_max_qvals.shape[0], target_max_qvals.shape[1], 1)
                #         # mix_q_target = mix_q_target.cuda()
                #         # _, rfm = th.sort(self.tindmix, descending=True)
                #         # for i in range(mix_qin_t):
                #         #     tar_max_qvals = self._target_mixer[rfm[0, i]](curr_counterfactual_actions_target_qvals,batch["state"][:, 1:])  #
                #         #     mix_q_target += tar_max_qvals
                #         # mix_q_target /= mix_qin_t
                #         # curr_counterfactual_actions_target_qvals = mix_q_target
                #         curr_counterfactual_actions_target_qvals = curr_counterfactual_actions_target_qvals.reshape(
                #             curr_counterfactual_actions_target_qvals.shape[0] * curr_counterfactual_actions_target_qvals.shape[1], 1)
                #
                #         counterfactual_actions_qvals.append(curr_counterfactual_actions_qvals)
                #         counterfactual_actions_target_qvals.append(curr_counterfactual_actions_target_qvals)
                #     # batch_size x episode_len, num_actions
                #     counterfactual_actions_qvals = th.cat(counterfactual_actions_qvals, 1)
                #     counterfactual_actions_target_qvals = th.cat(counterfactual_actions_target_qvals, 1)
                #
                #     all_counterfactual_actions_qvals.append(counterfactual_actions_qvals)
                #     all_counterfactual_actions_target_qvals.append(counterfactual_actions_target_qvals)
                #
                # # total_batch_size, num_agents, num_actions
                # all_counterfactual_actions_qvals = th.stack(all_counterfactual_actions_qvals).permute(1, 0, 2)
                # all_counterfactual_actions_target_qvals = th.stack(all_counterfactual_actions_target_qvals).permute(1, 0, 2)
                #
                # # total_batch_size, num_agents x num_actions
                # all_counterfactual_actions_qvals = all_counterfactual_actions_qvals.reshape(all_counterfactual_actions_qvals.shape[0], -1)
                # all_counterfactual_actions_target_qvals = all_counterfactual_actions_target_qvals.reshape(all_counterfactual_actions_target_qvals.shape[0], -1)
                #
                # softmax_weightings = self.softmax_weighting(all_counterfactual_actions_qvals)
                # softmax_qtots = softmax_weightings * all_counterfactual_actions_target_qvals
                # softmax_qtots = th.sum(softmax_qtots, 1, keepdim=True)
                #
                # softmax_qtots = softmax_qtots.reshape(rewards.shape[0], rewards.shape[1], rewards.shape[2])
                # target_max_qvals = softmax_qtots
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
                ts = self.target_mixer(target_max_qvals, batch["state"][:, 1:])  #
                target_max_qvals = mix_q_target
            # target_max_qvals = target_mac_out.max(dim=3)[0]
            # target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])
        #     target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        # if self.mixer is not None:
        #     chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
        #     mix_qind = th.zeros(1, self.args.km)
        #     for i in range(self.args.km):
        #         mix_qind[0, i] = (i + 1) * 1000000 / self.args.km
        #     for j in range(self.args.km):
        #         if t_env < mix_qind[0, j]:
        #             mix_qin_t = j + 1
        #             break
        #         else:
        #             mix_qin_t = self.args.km
        #     mix_q_target = th.zeros(target_max_qvals.shape[0], target_max_qvals.shape[1], 1)
        #     mix_q_target = mix_q_target.cuda()
        #     _, rfm = th.sort(self.tindmix, descending=True)
        #     for i in range(mix_qin_t):
        #         tar_max_qvals = self._target_mixer[rfm[0, i]](target_max_qvals, batch["state"][:, 1:])  #
        #         mix_q_target += tar_max_qvals
        #     mix_q_target /= mix_qin_t
        #     ts = self.target_mixer(target_max_qvals, batch["state"][:, 1:])  #
        #     target_max_qvals = mix_q_target
        # if self.mixer is not None:
        #     chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
        #     target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])
        td_lambda = self._td_lambda_target(batch, batch.max_seq_length - 1, target_max_qvals, self.args.n_agents, 0.99,0.8)
        td_lambda = td_lambda[:, :, 1].unsqueeze(2)
        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals
        # Td-error
        td_error = (chosen_action_qvals - targets.detach())
        mask = mask.expand_as(td_error)
        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask
        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        # q_return_diff = (chosen_action_qvals - td_lambda.detach())#.detach()
        # v_l2 = ((q_return_diff * mask) ** 2).sum() / mask.sum()
        # # loss += self.args.res_lambda * v_l2
        # loss += 0.5 * v_l2
        if self.args.res:
            future_episode_return = batch["future_discounted_return"][:, :-1]
            q_return_diff = (chosen_action_qvals - future_episode_return.detach())
            v_l2 = ((q_return_diff * mask) ** 2).sum() / mask.sum()
            loss += self.args.res_lambda * v_l2

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
            # self.idx += 1
            self.tindrnn[0, self.idx % self.args.k] = self.idx + 1
            self.idx += 1
            # print(self.tindrnn)
            # print(qin_t,rf)
            if self.args.km > 1:
                self._target_mixer[self.idxm % self.args.km].load_state_dict(self.mixer.state_dict())
            else:
                self._target_mixer.load_state_dict(self.mixer.state_dict())
            # self.idx += 1
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
        mask_elems = mask.sum().item()

        return np.mean((targets * mask).sum().item() / (mask_elems * self.args.n_agents)), np.mean(
            (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents)), \
               (masked_td_error.abs().sum().item() / mask_elems), grad_norm, loss.item()

    def traind(self, batch: EpisodeBatch, t_env: int, episode_num: int):
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
        targdd = []
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
                dd = self.target_mac.forward(batch, t=t)
                targdd.append(dd)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time
        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999
        # Max over target Q-Values
        # if self.args.double_q:
        #     # Get actions that maximise live Q (for double q-learning)
        #     mac_out_detach = mac_out.clone().detach()
        #     mac_out_detach[avail_actions == 0] = -9999999
        #     cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
        #     target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        # else:
        #     target_max_qvals = target_mac_out.max(dim=3)[0]
        # target_max_qvals = target_mac_out.max(dim=3)[0]
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            # mix_qind = th.zeros(1, self.args.km)
            # for i in range(self.args.km):
            #     mix_qind[0, i] = (i + 1) * self.qtot / self.args.km
            # for j in range(self.args.km):
            #     if t_env < mix_qind[0, j]:
            #         mix_qin_t = j + 1
            #         break
            #     else:
            #         mix_qin_t = self.args.km
            # 0 t_env % (self.args.trans*25) == 0  t_env > 1000000
            if t_env % (self.args.trans * 25) == 0:
                # c = th.max(target_mac_out,3).values.unsqueeze(3)
                # dd = th.exp(self.omega * (target_mac_out - c))
                # # rf = dd.reshape(-1, dd.shape[2],dd.shape[3])
                # df = th.sum(dd, -1, keepdim=True)/5
                # mellow = c + th.log(df) / self.omega
                # ans = th.sum(th.exp((target_mac_out - mellow) * self.beta) * (target_mac_out - mellow), -1, keepdim=True)
                # target_max_qvals = self.target_mixer(ans, batch["state"][:, 1:])
                dd = th.exp(self.omega * (target_mac_out))
                df = th.sum(dd, -1, keepdim=True) / 5
                mellow = th.log(df) / self.omega
                target_max_qvals = self.target_mixer(mellow, batch["state"][:, 1:])
                # mac_out_detach = mac_out.clone().detach()
                # mac_out_detach[avail_actions == 0] = -9999999
                # cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
                # target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
                # all_counterfactual_actions_qvals = []
                # all_counterfactual_actions_target_qvals = []
                # for agent_idx in range(cur_max_actions.shape[2]):
                #     base_actions = copy.deepcopy(cur_max_actions)
                #     # total_batch_size, num_agents
                #     base_actions = base_actions.squeeze(-1).reshape(-1, cur_max_actions.shape[2])
                #     # num_actions, 1
                #     all_actions_for_an_agent = th.tensor([action_idx for action_idx in range(self.args.n_actions)]).unsqueeze(0)
                #     # num_actions, total_batch_size: [[0, ..., 0], [1, ..., 1], ..., [4, ..., 4]]
                #     all_actions_for_an_agent = all_actions_for_an_agent.repeat(base_actions.shape[0], 1).transpose(1, 0)
                #     # formate to a column vector: total_batch_size x num_actions: [0, ..., 0, ...., 4, ..., 4]
                #     all_actions_for_an_agent = all_actions_for_an_agent.reshape(-1, 1).squeeze()
                #     # total_batch_size x num_agents, num_actions (repeat the actions for num_actions times)
                #     counterfactual_actions = base_actions.repeat(self.args.n_actions, 1).reshape(-1, base_actions.shape[1])
                #     counterfactual_actions[:, agent_idx] = all_actions_for_an_agent
                #     counterfactual_actions_qvals, counterfactual_actions_target_qvals = [], []
                #     for action_idx in range(self.args.n_actions):
                #         curr_counterfactual_actions = counterfactual_actions[action_idx * base_actions.shape[0] : (action_idx + 1) * base_actions.shape[0]]
                #         curr_counterfactual_actions = curr_counterfactual_actions.reshape(cur_max_actions.shape[0], cur_max_actions.shape[1], cur_max_actions.shape[2], -1)
                #         # batch_size, episode_len, num_agents
                #         curr_counterfactual_actions_qvals = th.gather(mac_out_detach[:, 1:], dim=3, index=curr_counterfactual_actions).squeeze(3)  # Remove the last dim
                #         curr_counterfactual_actions_target_qvals = th.gather(target_mac_out, dim=3, index=curr_counterfactual_actions).squeeze(3)  # Remove the last dim
                #         # batch_size, episode_len, 1
                #         curr_counterfactual_actions_qvals = self.mixer(curr_counterfactual_actions_qvals, batch["state"][:, 1:])
                #         curr_counterfactual_actions_qvals = curr_counterfactual_actions_qvals.reshape(curr_counterfactual_actions_qvals.shape[0] * curr_counterfactual_actions_qvals.shape[1], 1)
                #
                #         curr_counterfactual_actions_target_qvals = self.target_mixer(curr_counterfactual_actions_target_qvals, batch["state"][:, 1:])
                #         # mix_q_target = th.zeros(target_max_qvals.shape[0], target_max_qvals.shape[1], 1)
                #         # mix_q_target = mix_q_target.cuda()
                #         # _, rfm = th.sort(self.tindmix, descending=True)
                #         # for i in range(mix_qin_t):
                #         #     tar_max_qvals = self._target_mixer[rfm[0, i]](curr_counterfactual_actions_target_qvals,batch["state"][:, 1:])  #
                #         #     mix_q_target += tar_max_qvals
                #         # mix_q_target /= mix_qin_t
                #         # curr_counterfactual_actions_target_qvals = mix_q_target
                #         curr_counterfactual_actions_target_qvals = curr_counterfactual_actions_target_qvals.reshape(
                #             curr_counterfactual_actions_target_qvals.shape[0] * curr_counterfactual_actions_target_qvals.shape[1], 1)
                #
                #         counterfactual_actions_qvals.append(curr_counterfactual_actions_qvals)
                #         counterfactual_actions_target_qvals.append(curr_counterfactual_actions_target_qvals)
                #     # batch_size x episode_len, num_actions
                #     counterfactual_actions_qvals = th.cat(counterfactual_actions_qvals, 1)
                #     counterfactual_actions_target_qvals = th.cat(counterfactual_actions_target_qvals, 1)
                #
                #     all_counterfactual_actions_qvals.append(counterfactual_actions_qvals)
                #     all_counterfactual_actions_target_qvals.append(counterfactual_actions_target_qvals)
                #
                # # total_batch_size, num_agents, num_actions
                # all_counterfactual_actions_qvals = th.stack(all_counterfactual_actions_qvals).permute(1, 0, 2)
                # all_counterfactual_actions_target_qvals = th.stack(all_counterfactual_actions_target_qvals).permute(1, 0, 2)
                #
                # # total_batch_size, num_agents x num_actions
                # all_counterfactual_actions_qvals = all_counterfactual_actions_qvals.reshape(all_counterfactual_actions_qvals.shape[0], -1)
                # all_counterfactual_actions_target_qvals = all_counterfactual_actions_target_qvals.reshape(all_counterfactual_actions_target_qvals.shape[0], -1)
                #
                # softmax_weightings = self.softmax_weighting(all_counterfactual_actions_qvals)
                # softmax_qtots = softmax_weightings * all_counterfactual_actions_target_qvals
                # softmax_qtots = th.sum(softmax_qtots, 1, keepdim=True)
                #
                # softmax_qtots = softmax_qtots.reshape(rewards.shape[0], rewards.shape[1], rewards.shape[2])
                # target_max_qvals = softmax_qtots
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
                ts = self.target_mixer(target_max_qvals, batch["state"][:, 1:])  #
                target_max_qvals = mix_q_target

        # Calculate 1-step Q-Learning targets
        td_lambda = self._td_lambda_target(batch, batch.max_seq_length - 1, target_max_qvals, self.args.n_agents, 0.99,0.8)
        td_lambda = td_lambda[:, :, 1].unsqueeze(2)
        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals
        # Td-error
        td_error = (chosen_action_qvals - targets.detach())
        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask
        loss = (masked_td_error ** 2).sum() / mask.sum()
        if self.args.res:
            future_episode_return = batch["future_discounted_return"][:, :-1]
            q_return_diff = (chosen_action_qvals - future_episode_return.detach())
            v_l2 = ((q_return_diff * mask) ** 2).sum() / mask.sum()
            loss += self.args.res_lambda * v_l2

        mask_elems = mask.sum().item()

        future_episode_return = batch["future_discounted_return"][:, :-1]
        future_episode_return = future_episode_return.detach()

        ture_returns = th.mean(future_episode_return)
        estimate_returns = (target_max_qvals * mask).sum().item() / (mask_elems * self.args.n_agents)

        import random
        index1 = random.sample(range(20), 10)
        indices = th.tensor(index1)
        # indices = indices.cuda()
        b = th.index_select(future_episode_return, 0, indices)
        b1 = th.index_select(target_max_qvals, 0, indices)
        b2 = th.index_select(td_lambda, 0, indices)
        index2 = random.sample(range(25), 10)
        indic = th.tensor(index2)
        # indic = indic.cuda()

        d = th.index_select(b, 1, indic)
        d1 = th.index_select(b1, 1, indic)
        d2 = th.index_select(b2, 1, indic)
        ture_returns150 = th.mean(d)
        estimate_returns150 = th.mean(d1)
        td_lambda150 = th.mean(d2)
        return ture_returns150, estimate_returns150, ture_returns, estimate_returns, td_lambda150, np.mean(
            (targets * mask).sum().item() / (mask_elems * self.args.n_agents)), np.mean(
            (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents)), \
               (masked_td_error.abs().sum().item() / mask_elems), loss.item()

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

    def _td_lambda_target(self, batch, max_episode_len, q_targets, n_agents, gamma, td_lambda):
        # batch.shep = (episode_num, max_episode_len， n_agents，n_actions)
        # q_targets.shape = (episode_num, max_episode_len， n_agents)
        # mask = (1 - batch["padded"].float()).repeat(1, 1, n_agents)
        episode_num = batch["reward"].shape[0]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        mask = mask.repeat(1, 1, n_agents)
        terminated = (1 - batch["terminated"][:, :-1].float()).repeat(1, 1, n_agents)
        rewas = batch["reward"][:, :-1]
        r = rewas.repeat((1, 1, n_agents))

        # --------------------------------------------------n_step_return---------------------------------------------------
        '''
        1. 每条经验都有若干个n_step_return，所以给一个最大的max_episode_len维度用来装n_step_return
        最后一维,第n个数代表 n+1 step。
        2. 因为batch中各个episode的长度不一样，所以需要用mask将多出的n-step return置为0，
        否则的话会影响后面的lambda return。第t条经验的lambda return是和它后面的所有n-step return有关的，
        如果没有置0，在计算td-error后再置0是来不及的
        3. terminated用来将超出当前episode长度的q_targets和r置为0
        '''
        n_step_return = th.zeros((episode_num, max_episode_len, n_agents, max_episode_len))
        # n_step_return = n_step_return.cuda()
        for transition_idx in range(max_episode_len - 1, -1, -1):
            # 最后计算1 step return
            n_step_return[:, transition_idx, :, 0] = (r[:, transition_idx] + gamma * q_targets[:, transition_idx] *
                                                      terminated[:, transition_idx]) * mask[:,
                                                                                       transition_idx]  # 经验transition_idx上的obs有max_episode_len - transition_idx个return, 分别计算每种step return
            # 同时要注意n step return对应的index为n-1
            for n in range(1, max_episode_len - transition_idx):
                # t时刻的n step return =r + gamma * (t + 1 时刻的 n-1 step return)
                # n=1除外, 1 step return =r + gamma * (t + 1 时刻的 Q)
                n_step_return[:, transition_idx, :, n] = (r[:, transition_idx] + gamma * n_step_return[:,
                                                                                         transition_idx + 1, :,
                                                                                         n - 1]) * mask[:,
                                                                                                   transition_idx]
        # --------------------------------------------------n_step_return---------------------------------------------------

        # --------------------------------------------------lambda return---------------------------------------------------
        '''
        lambda_return.shape = (episode_num, max_episode_len，n_agents)
        '''
        lambda_return = th.zeros((episode_num, max_episode_len, n_agents))
        # lambda_return = lambda_return.cuda()
        for transition_idx in range(max_episode_len):
            returns = th.zeros((episode_num, n_agents))
            # returns = returns.cuda()
            for n in range(1, max_episode_len - transition_idx):
                returns += pow(td_lambda, n - 1) * n_step_return[:, transition_idx, :, n - 1]
            lambda_return[:, transition_idx] = (1 - td_lambda) * returns + \
                                               pow(td_lambda, max_episode_len - transition_idx - 1) * \
                                               n_step_return[:, transition_idx, :, max_episode_len - transition_idx - 1]
        # --------------------------------------------------lambda return---------------------------------------------------
        return lambda_return

    def _build_td_lambda_targets(self, batch, target_qs, n_agents, gamma, td_lambda):
        # Assumes  <target_qs > in B*T*A and <reward >, <terminated >, <mask > in (at least) B*T-1*1
        # Initialise  last  lambda -return  for  not  terminated  episodes
        rewards = batch["reward"]
        terminated = batch["terminated"].float()
        mask = batch["filled"].float()
        mask = mask * (1 - terminated)

        ret = target_qs.new_zeros(*target_qs.shape)
        ret[:, -1] = target_qs[:, -1] * (1 - th.sum(terminated, dim=1))
        # Backwards  recursive  update  of the "forward  view"
        for t in range(ret.shape[1] - 2, -1, -1):
            ret[:, t] = td_lambda * gamma * ret[:, t + 1] + mask[:, t] \
                        * (rewards[:, t] + (1 - td_lambda) * gamma * target_qs[:, t + 1] * (1 - terminated[:, t]))
        # Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*A
        return ret[:, 0:-1]
