import numpy as np
import torch
import torch.optim as optim
from torch.nn import utils
from torch import nn
import random
import pandas as pd
import csv
from random import choice
from torch.nn import functional as F
import os, glob
from torch.utils.data import Dataset, DataLoader
import visdom

device = "cuda:0"


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)


def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()


num_timesteps = 500000
n_iter = 500000
obs_dim = 149
ac_dim = 4
double_q = True
grad_norm_clipping = 10
gamma = 1.01
learning_starts = 5000
learning_freq = 500
target_update_freq = 2000
replay_buffer_size = 50000
frame_history_len = 1
lr = 5e-4
batchsz = 3900
num_agent_train_steps_per_iter = 5
print_period = 1000

class Read_data(Dataset):

    def __init__(self, root,  mode):
        """
        :param root: the path of the dataset
        :param resize: the shape of the signal
        :param mode: the use of the dataset (train / validatation / test)
        """
        super(Read_data, self).__init__()

        self.root = root

        # to initialize the label to the signal
        self.name2label = {}
        for name in sorted(os.listdir(root)):
        # os.listdir(): to get the name of the file of the path given
            if not os.path.isdir(os.path.join(root, name)):
            # os.path.isdir(): to decide whether the certain root is a file folder
            # os.path.join( , ): to concatenate the path
                continue

            self.name2label[name] = len(self.name2label.keys())
            # set the label to the data

        # to get the signals and the labels
        self.signals, self.labels = self.load_csv('signals.csv')

        # to split the dataset
        if mode == 'train': # 60%
            self.signals = self.signals[:int(0.6 * len(self.signals))]
            self.labels = self.labels[:int(0.6 * len(self.labels))]
        elif mode == 'val': # 20% = 60% -> 80%
            self.signals = self.signals[int(0.6 * len(self.signals)):int(0.8 * len(self.signals))]
            self.labels = self.labels[int(0.6 * len(self.labels)):int(0.8 * len(self.labels))]
        else: # 20% = 80% -> end
            self.signals = self.signals[int(0.8 * len(self.signals)):]
            self.labels = self.labels[int(0.8 * len(self.labels)):]

    def load_csv(self, filename):

        if not os.path.exists(os.path.join(self.root, filename)):
        # os.path.exists(): to check if there is the  certain file
            # to create the csv file
            signals = []
            for name in self.name2label.keys():
                for i in range(51):
                    signals += glob.glob(os.path.join(self.root, name, str(i - 20) + 'dB', '*.csv'))

            random.shuffle(signals)
            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for sig in signals:
                    name = sig.split(os.sep)[-3]
                    # os.sep: the break like '/' in MAC OS operation system
                    label = self.name2label[name]
                    writer.writerow([sig, label])
                print("write into csv file:", filename)

        # read the csv file
        signals, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                sig, label = row
                label = int(label)
                signals.append(sig)
                labels.append(label)

        assert len(signals) == len(labels)

        return signals, labels

    def __len__(self):
        # this function enable we use len() to get the length of the dataset
        return len(self.signals)

    def __getitem__(self, idx):
        # this function enable we use p[key] to get the value
        if idx < 0 or idx > len(self.signals) - 1:
            print("the idx is wrong!")
            os._exit(1)
        sig, label = self.signals[idx], self.labels[idx]

        data = torch.from_numpy(pd.read_csv(sig).values).float()
        label = torch.tensor(label)

        return data, label


train_db = Read_data('data', mode='train')
val_db = Read_data('data', mode='val')

# set up a DataLoader object
train_loader = DataLoader(train_db, batch_size=int(0.6*3000*26*4), shuffle=True)
val_loader = DataLoader(val_db, batch_size=int(0.2*3000*26*4), shuffle=False)
for _, (x, y) in enumerate(train_loader):
    train_data_x, train_data_y = x, y
for _, (x, y) in enumerate(val_loader):
    val_data_x, val_data_y = x, y

class Net(nn.Module):

    def __init__(self, num_class):
        super(Net, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(2, 3, kernel_size=3, stride=2, padding=0),
            # nn.BatchNorm1d(3),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=3, stride=2, padding=0),
            # nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=3, stride=2, padding=0),
            # nn.BatchNorm1d(16)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=0),
            # nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=0),
            # nn.BatchNorm1d(32)
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=0),
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=0),
            # nn.BatchNorm1d(64)
        )
        self.line1 = nn.Linear(64*7, 256)
        # self.bn1 = nn.BatchNorm1d(256)
        self.line2 = nn.Linear(256, 256)
        self.dp2 = nn.Dropout(0.5)
        self.line3 = nn.Linear(256, 256)
        self.dp3 = nn.Dropout(0.5)
        self.line4 = nn.Linear(256, 128)
        self.dp4 = nn.Dropout(0.5)
        self.line5 = nn.Linear(128, 32)
        self.dp5 = nn.Dropout(0.5)
        self.out = nn.Linear(32, num_class)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        # x = x.reshape(-1, 2, 1023)
        x = F.relu(self.conv1(x))
        # x = x.squeeze()
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # x = F.adaptive_max_pool2d(x, [1, 1])

        # flaten operation
        x = x.view(x.size(0), -1)
        # [b, 32*3*3] => [b, 10]
        # print(x.shape)
        x = F.relu(self.line1(x))
        x = F.relu(self.dp2(self.line2(x)))
        x = F.relu(self.dp3(self.line3(x)))
        x = F.relu(self.dp4(self.line4(x)))
        x = F.relu(self.dp5(self.line5(x)))
        x = self.out(x)

        return x

def create_lander_q_network():
    return Net(4)


class Environment(object):

    def __init__(self, data_x, data_y, model=None):
        self.train_X = data_x
        self.train_Y = data_y
        self.current_index = self._sample_index()
        self.action_space = ac_dim
        self.model = model

    def reset(self):
        """
        :return: random x from dataset
        """
        obs, _ = self.step(-1)
        return obs

    def step(self, action):
        if action == -1:
            _c_index = self.current_index
            # self.current_index = self._sample_index()
            return (self.train_X[_c_index].transpose(1, 0).unsqueeze(0), 0)

        r = self.reward(action)
        self.current_index = self._sample_index()
        return self.train_X[self.current_index].transpose(1, 0).unsqueeze(0), r, 0

    def reward(self, action):
        c = self.train_Y[self.current_index]
        return 1 if c == action else -2

    def sample_actions(self):
        return random.randint(0, self.action_space - 1)

    def _sample_index(self):
        if not ('self.current_index' in locals().keys()):
            return random.randint(0, len(self.train_Y) - 1)
        else:
            ob = self.train_X[self.current_index]
            ob = from_numpy(ob)
            actions = self.model(ob)
            _, ind = torch.sort(actions)
            indx_second = ind[-2].item()
            x = choice(self.train_X[indx_second == self.train_Y])
            arr = np.sum(self.train_X - x, 1)
            ind_ = np.where(arr == 0)[0].item()

            return ind_


class ArgMaxPolicy(object):

    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs):
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]
        actions = self.critic.qa_values(obs)
        action = np.argmax(actions)

        return action.squeeze()


class DQNCritic():

    def __init__(self):
        super().__init__()
        self.ob_dim = 149

        if isinstance(self.ob_dim, int):
            self.input_shape = (self.ob_dim,)

        self.ac_dim = ac_dim
        self.double_q = double_q
        self.grad_norm_clipping = grad_norm_clipping
        self.gamma = gamma

        self.q_net = create_lander_q_network()
        self.q_net_target = create_lander_q_network()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss = nn.SmoothL1Loss()  # AKA Huber loss
        self.q_net.to(device)
        self.q_net_target.to(device)

    def update(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n):
        ob_no = from_numpy(ob_no)
        ac_na = from_numpy(ac_na).to(torch.long)
        # ob_no = pgd_attack(self.q_net_target, ob_no, ac
        next_ob_no = from_numpy(next_ob_no)
        # next_ob_no = pgd_attack(self.q_net_target, next_ob_no, ac_na)
        reward_n = from_numpy(reward_n)
        terminal_n = from_numpy(terminal_n)
        ob_no = ob_no.reshape(-1, 2, 1023)
        next_ob_no = next_ob_no.reshape(-1, 2, 1023)
        qa_t_values = self.q_net(ob_no)
        q_t_values = torch.gather(qa_t_values, 1, ac_na.unsqueeze(1)).squeeze(1)

        # TODO compute the Q-values from the target network
        qa_tp1_values = self.q_net_target(next_ob_no)

        if self.double_q:
            q_tp1 = torch.gather(qa_tp1_values, 1, torch.argmax(qa_t_values, 1, keepdim=True)).squeeze(1)

        else:
            q_tp1, _ = qa_tp1_values.max(dim=1)

        target = reward_n + self.gamma * q_tp1
        target = target.detach()

        assert q_t_values.shape == target.shape
        loss = self.loss(q_t_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_value_(self.q_net.parameters(), self.grad_norm_clipping)
        self.optimizer.step()

        return {
            'Training Loss': to_numpy(loss),
        }

    def update_target_network(self):
        for target_param, param in zip(
                self.q_net_target.parameters(), self.q_net.parameters()
        ):
            target_param.data.copy_(param.data)

    def qa_values(self, obs):
        obs = from_numpy(obs)
        qa_values = self.q_net(obs)
        return to_numpy(qa_values)


class MemoryOptimizedReplayBuffer(object):
    def __init__(self, size, frame_history_len, lander=False):
        self.lander = lander

        self.size = size
        self.frame_history_len = frame_history_len

        self.next_idx = 0
        self.num_in_buffer = 0

        self.obs = None
        self.action = None
        self.reward = None
        self.done = None

    def can_sample(self, batch_size):
        return batch_size + 1 <= self.num_in_buffer

    def _encode_sample(self, idxes):
        obs_batch = np.concatenate([self._encode_observation(idx)[None] for idx in idxes], 0)
        act_batch = self.action[idxes]
        rew_batch = self.reward[idxes]
        next_obs_batch = np.concatenate([self._encode_observation(idx + 1)[None] for idx in idxes], 0)
        done_mask = np.array([1.0 if self.done[idx] else 0.0 for idx in idxes], dtype=np.float32)

        return obs_batch, act_batch, rew_batch, next_obs_batch, done_mask

    def sample_n_unique(self, sampling_f, n):
        res = []
        while len(res) < n:
            candidate = sampling_f()
            if candidate not in res:
                res.append(candidate)
        return res

    def sample(self, batch_size):
        assert self.can_sample(batch_size)
        idxes = self.sample_n_unique(lambda: random.randint(0, self.num_in_buffer - 2), batch_size)
        return self._encode_sample(idxes)

    def encode_recent_observation(self):

        assert self.num_in_buffer > 0
        return self._encode_observation((self.next_idx - 1) % self.size)

    def _encode_observation(self, idx):
        end_idx = idx + 1  # make noninclusive
        start_idx = end_idx - self.frame_history_len
        if len(self.obs.shape) == 2:
            return self.obs[end_idx - 1]
        if start_idx < 0 and self.num_in_buffer != self.size:
            start_idx = 0
        for idx in range(start_idx, end_idx - 1):
            if self.done[idx % self.size]:
                start_idx = idx + 1
        missing_context = self.frame_history_len - (end_idx - start_idx)
        if start_idx < 0 or missing_context > 0:
            frames = [np.zeros_like(self.obs[0]) for _ in range(missing_context)]
            for idx in range(start_idx, end_idx):
                frames.append(self.obs[idx % self.size])
            return np.concatenate(frames, 0)
        else:
            img_h, img_w = self.obs.shape[2], self.obs.shape[3]
            return self.obs[start_idx:end_idx].reshape(-1, img_h, img_w)

    def store_frame(self, frame):

        if self.obs is None:
            self.obs = np.empty([self.size] + list(frame.shape), dtype=np.float32 if self.lander else np.uint8)
            self.action = np.empty([self.size], dtype=np.int32)
            self.reward = np.empty([self.size], dtype=np.float32)
            self.done = np.empty([self.size], dtype=np.bool)
        self.obs[self.next_idx] = frame

        ret = self.next_idx
        self.next_idx = (self.next_idx + 1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)

        return ret

    def store_effect(self, idx, action, reward, done):
        self.action[idx] = action
        self.reward[idx] = reward
        self.done[idx] = done


def linear_interpolation(l, r, alpha):
    return l + alpha * (r - l)


class PiecewiseSchedule(object):
    def __init__(self, endpoints, interpolation=linear_interpolation, outside_value=None):

        idxes = [e[0] for e in endpoints]
        assert idxes == sorted(idxes)
        self._interpolation = interpolation
        self._outside_value = outside_value
        self._endpoints = endpoints

    def value(self, t):
        for (l_t, l), (r_t, r) in zip(self._endpoints[:-1], self._endpoints[1:]):
            if l_t <= t and t < r_t:
                alpha = float(t - l_t) / (r_t - l_t)
                return self._interpolation(l, r, alpha)
        assert self._outside_value is not None
        return self._outside_value


def lander_exploration_schedule(num_timesteps):
    return PiecewiseSchedule(
        [
            (0, 1),
            (num_timesteps * 0.1, 0.02),
        ], outside_value=0.02
    )


class DQNAgent(object):
    def __init__(self):

        self.critic = DQNCritic()
        self.env = Environment(train_data_x, train_data_y, self.critic.q_net)
        self.batch_size = batchsz
        self.last_obs = self.env.reset()

        self.num_actions = ac_dim
        self.learning_starts = learning_starts
        self.learning_freq = learning_freq
        self.target_update_freq = target_update_freq

        self.replay_buffer_idx = None
        self.exploration = lander_exploration_schedule(num_timesteps)


        self.actor = ArgMaxPolicy(self.critic)

        self.replay_buffer = MemoryOptimizedReplayBuffer(
            replay_buffer_size, frame_history_len)
        self.t = 0
        self.num_param_updates = 0

    def add_to_replay_buffer(self, paths):
        pass

    def step_env(self):
        self.replay_buffer_idx = self.replay_buffer.store_frame(self.last_obs)

        eps = self.exploration.value(self.t)

        perform_random_action = eps >= np.random.randn()
        if perform_random_action:
            action = np.random.randint(0, self.num_actions)
        else:
            action = self.actor.get_action(self.replay_buffer.encode_recent_observation())

        obs, reward, done = self.env.step(action)
        self.last_obs = obs

        self.replay_buffer.store_effect(self.replay_buffer_idx, action, reward, done)

        if done:
            self.last_obs = self.env.reset()

    def sample(self, batch_size):
        if self.replay_buffer.can_sample(self.batch_size):
            return self.replay_buffer.sample(batch_size)
        else:
            return [], [], [], [], []

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        log = {}
        if (self.t > self.learning_starts
                and self.t % self.learning_freq == 0
                and self.replay_buffer.can_sample(self.batch_size)
        ):

            log = self.critic.update(
                ob_no, ac_na, next_ob_no, re_n, terminal_n  # need to check the shape of the tensor
            )

            if self.num_param_updates % self.target_update_freq == 0:
                self.critic.update_target_network()

            self.num_param_updates += 1

        self.t += 1
        return log

    def evaluate(self):
        correct = 0
        total = len(val_data_y)
        X = val_data_x
        Y = val_data_y
        x, y = np.array(X), np.array(Y)
        x, y = from_numpy(x), from_numpy(y)
        x = x.permute(0, 2, 1)
        with torch.no_grad():
            logits = self.critic.q_net(x)
            # print(logits.shape)
            pred = logits.argmax(dim=1)
        correct = torch.eq(pred, y).sum().float().item()

        return correct / total


def train_agent(agent):
    all_logs = []
    for train_step in range(num_agent_train_steps_per_iter):
        ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = agent.sample(batchsz)

        train_log = agent.train(ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch)
        all_logs.append(train_log)
    return all_logs


def pgd_attack(model, images, labels, eps=0.3, alpha=1 / 255, iters=30):
    mi = torch.min(images)
    ma = torch.max(images)
    loss_pgd = nn.CrossEntropyLoss()
    ori_images = images.detach()

    for i in range(iters):
        images.requires_grad = True
        images = images.reshape(-1, 2, 1023)
        labels = labels.reshape(-1, 1)
        outputs = model(images)

        model.zero_grad()
        cost = loss_pgd(outputs, labels).to(device)
        cost.backward()

        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=mi, max=ma).detach_()

    return images

viz = visdom.Visdom()
def run_training_loop(n_iter):
    """
    :param n_iter:  number of (dagger) iterations
    :param collect_policy:
    :param eval_policy:
    """

    # init vars at beginning of training
    total_envsteps = 0
    agent = DQNAgent()
    acc_max = 0
    for itr in range(n_iter):
        if itr % print_period == 0:
            print("\n\n********** Iteration %i ************" % itr)
            acc = agent.evaluate()
            if acc > acc_max:
                acc_max = acc
            viz.line([[float(acc)]], [itr], win='acc_rl',
                     opts=dict(title='acc_rl', legend=['acc']), update='append')
            print("acc: ", acc, " max_acc: ", acc_max)

        # collect trajectories, to be used for training
        # only perform an env step and add to replay buffer for DQN
        agent.step_env()
        envsteps_this_batch = 1

        total_envsteps += envsteps_this_batch

        # train agent (using sampled data from replay buffer)
        if itr % print_period == 0:
            print("\nTraining agent...")
        all_logs = train_agent(agent)


def main():
    run_training_loop(n_iter)


if __name__ == "__main__":
    main()