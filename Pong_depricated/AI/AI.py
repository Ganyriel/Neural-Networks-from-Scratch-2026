import gymnasium as gym
import numpy as np
from collections import deque
import ale_py

gym.register_envs(ale_py)


# ============================================================
# Configuration
# ============================================================

ENV_ID = "ALE/Pong-v5"

SEED = 42

# Atari preprocessing
IMAGE_SIZE = 84
FRAME_STACK = 4
FRAME_SKIP = 4

# DQN
GAMMA = 0.99
LEARNING_RATE = 2.5e-4

REPLAY_SIZE = 100_000
BATCH_SIZE = 32

LEARNING_STARTS = 20_000
TRAIN_EVERY = 4
TARGET_UPDATE_EVERY = 10_000

# Exploration
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY_STEPS = 1_000_000

# Training
MAX_STEPS = 5_000_000

# Gradient clipping
MAX_GRAD_NORM = 10.0

MODEL_FILE = "pong_dqn.npz"


# ============================================================
# Random number generator
# ============================================================

rng = np.random.default_rng(SEED)


# ============================================================
# Frame preprocessing
# ============================================================

def preprocess_frame(frame):
    """
    Convert Atari RGB frame:

        (210, 160, 3)

    into:

        (84, 84)

    using only NumPy.

    We crop the Atari score area and resize using nearest-neighbor.
    """

    # Crop the top/bottom borders.
    #
    # Atari Pong has useful game information roughly in this region.
    frame = frame[34:194, :, :]

    # RGB -> grayscale.
    #
    # This avoids needing PIL/OpenCV.
    gray = (
        0.299 * frame[:, :, 0]
        + 0.587 * frame[:, :, 1]
        + 0.114 * frame[:, :, 2]
    )

    gray = gray.astype(np.uint8)

    # Nearest-neighbor resize to 84x84.
    h, w = gray.shape

    y_indices = np.linspace(0, h - 1, IMAGE_SIZE).astype(np.int32)
    x_indices = np.linspace(0, w - 1, IMAGE_SIZE).astype(np.int32)

    resized = gray[y_indices[:, None], x_indices[None, :]]

    return resized


# ============================================================
# Frame stack
# ============================================================

class FrameStack:
    def __init__(self, num_frames):
        self.num_frames = num_frames
        self.frames = deque(maxlen=num_frames)

    def reset(self, frame):
        self.frames.clear()

        for _ in range(self.num_frames):
            self.frames.append(frame.copy())

        return self.get()

    def append(self, frame):
        self.frames.append(frame.copy())
        return self.get()

    def get(self):
        return np.stack(self.frames, axis=0)


# ============================================================
# Atari environment wrapper
# ============================================================

class PongEnv:
    """
    Handles:

        - preprocessing
        - frame skipping
        - frame stacking
        - reward accumulation
    """

    def __init__(self):
        # We explicitly set frameskip=1 because we implement
        # frame skipping ourselves below.
        #
        # This also gives us control over the exact frame
        # that gets stored in the stack.

        self.env = gym.make(
            ENV_ID,
            frameskip=1,
            repeat_action_probability=0.0,
            full_action_space=False,
        )

        self.frame_stack = FrameStack(FRAME_STACK)

        self.num_actions = self.env.action_space.n

    def reset(self, seed=None):

        obs, info = self.env.reset(seed=seed)

        frame = preprocess_frame(obs)

        state = self.frame_stack.reset(frame)

        return state, info

    def step(self, action):

        total_reward = 0.0
        terminated = False
        truncated = False

        last_frame = None

        # -----------------------------
        # Frame skipping
        # -----------------------------

        for _ in range(FRAME_SKIP):

            obs, reward, terminated, truncated, info = \
                self.env.step(action)

            total_reward += reward

            last_frame = obs

            if terminated or truncated:
                break

        # Only store the final frame from the skipped sequence.
        frame = preprocess_frame(last_frame)

        next_state = self.frame_stack.append(frame)

        return (
            next_state,
            total_reward,
            terminated,
            truncated,
            info,
        )

    def close(self):
        self.env.close()


# ============================================================
# Replay buffer
# ============================================================

class ReplayBuffer:

    def __init__(self, capacity):

        self.capacity = capacity

        self.states = np.empty(
            (capacity, FRAME_STACK, IMAGE_SIZE, IMAGE_SIZE),
            dtype=np.uint8,
        )

        self.next_states = np.empty(
            (capacity, FRAME_STACK, IMAGE_SIZE, IMAGE_SIZE),
            dtype=np.uint8,
        )

        self.actions = np.empty(
            capacity,
            dtype=np.int64,
        )

        self.rewards = np.empty(
            capacity,
            dtype=np.float32,
        )

        self.dones = np.empty(
            capacity,
            dtype=np.float32,
        )

        self.position = 0
        self.size = 0

    def add(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):

        i = self.position

        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = done

        self.position = (self.position + 1) % self.capacity

        self.size = min(
            self.size + 1,
            self.capacity,
        )

    def sample(self, batch_size):

        indices = rng.integers(
            0,
            self.size,
            size=batch_size,
        )

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )

    def __len__(self):
        return self.size


# ============================================================
# NumPy neural network
# ============================================================

class DQN:

    """
    A small fully-connected DQN.

    Input:
        4 x 84 x 84 = 28,224 values

    Network:

        28224 -> 256 -> 256 -> actions

    This is deliberately simple so everything can be implemented
    directly in NumPy.

    Note:
        A convolutional DQN is substantially better for Atari,
        but implementing convolution + backprop entirely in NumPy
        would make this example much larger and significantly slower.
    """

    def __init__(self, num_actions):

        self.num_actions = num_actions

        input_size = FRAME_STACK * IMAGE_SIZE * IMAGE_SIZE

        hidden1 = 256
        hidden2 = 256

        # He initialization.

        self.W1 = (
            rng.standard_normal(
                (input_size, hidden1)
            ).astype(np.float32)
            * np.sqrt(2.0 / input_size)
        )

        self.b1 = np.zeros(
            hidden1,
            dtype=np.float32,
        )

        self.W2 = (
            rng.standard_normal(
                (hidden1, hidden2)
            ).astype(np.float32)
            * np.sqrt(2.0 / hidden1)
        )

        self.b2 = np.zeros(
            hidden2,
            dtype=np.float32,
        )

        self.W3 = (
            rng.standard_normal(
                (hidden2, num_actions)
            ).astype(np.float32)
            * np.sqrt(2.0 / hidden2)
        )

        self.b3 = np.zeros(
            num_actions,
            dtype=np.float32,
        )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    def forward(self, states):

        # Normalize pixels.
        x = states.astype(np.float32) / 255.0

        # Flatten.
        x = x.reshape(x.shape[0], -1)

        z1 = x @ self.W1 + self.b1
        a1 = np.maximum(z1, 0.0)

        z2 = a1 @ self.W2 + self.b2
        a2 = np.maximum(z2, 0.0)

        q = a2 @ self.W3 + self.b3

        cache = (
            x,
            z1,
            a1,
            z2,
            a2,
        )

        return q, cache

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    def predict(self, states):

        q, _ = self.forward(states)

        return q

    # --------------------------------------------------------
    # Copy parameters
    # --------------------------------------------------------

    def copy_from(self, other):

        self.W1 = other.W1.copy()
        self.b1 = other.b1.copy()

        self.W2 = other.W2.copy()
        self.b2 = other.b2.copy()

        self.W3 = other.W3.copy()
        self.b3 = other.b3.copy()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    def save(self, filename):

        np.savez(
            filename,
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=self.b2,
            W3=self.W3,
            b3=self.b3,
        )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    def load(self, filename):

        data = np.load(filename)

        self.W1[:] = data["W1"]
        self.b1[:] = data["b1"]

        self.W2[:] = data["W2"]
        self.b2[:] = data["b2"]

        self.W3[:] = data["W3"]
        self.b3[:] = data["b3"]


# ============================================================
# Adam optimizer
# ============================================================

class Adam:

    def __init__(
        self,
        network,
        learning_rate=2.5e-4,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
    ):

        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.t = 0

        self.parameters = [
            network.W1,
            network.b1,
            network.W2,
            network.b2,
            network.W3,
            network.b3,
        ]

        self.m = [
            np.zeros_like(p)
            for p in self.parameters
        ]

        self.v = [
            np.zeros_like(p)
            for p in self.parameters
        ]

    def step(self, gradients):

        self.t += 1

        for i, (p, g) in enumerate(
            zip(self.parameters, gradients)
        ):

            self.m[i] = (
                self.beta1 * self.m[i]
                + (1.0 - self.beta1) * g
            )

            self.v[i] = (
                self.beta2 * self.v[i]
                + (1.0 - self.beta2) * (g * g)
            )

            m_hat = (
                self.m[i]
                / (1.0 - self.beta1 ** self.t)
            )

            v_hat = (
                self.v[i]
                / (1.0 - self.beta2 ** self.t)
            )

            p -= (
                self.lr
                * m_hat
                / (np.sqrt(v_hat) + self.epsilon)
            )


# ============================================================
# Huber loss
# ============================================================

def huber_loss_and_gradient(error, delta=1.0):

    abs_error = np.abs(error)

    quadratic = abs_error <= delta

    loss = np.where(
        quadratic,
        0.5 * error * error,
        delta * (
            abs_error - 0.5 * delta
        ),
    )

    gradient = np.where(
        quadratic,
        error,
        delta * np.sign(error),
    )

    return loss, gradient


# ============================================================
# One DQN training step
# ============================================================

def train_step(
    online,
    target,
    optimizer,
    replay,
):

    states, actions, rewards, next_states, dones = \
        replay.sample(BATCH_SIZE)

    # --------------------------------------------------------
    # Current Q values
    # --------------------------------------------------------

    q_values, cache = online.forward(states)

    batch_indices = np.arange(BATCH_SIZE)

    chosen_q = q_values[
        batch_indices,
        actions,
    ]

    # --------------------------------------------------------
    # Target Q values
    # --------------------------------------------------------

    next_q = target.predict(next_states)

    max_next_q = np.max(
        next_q,
        axis=1,
    )

    targets = (
        rewards
        + GAMMA
        * (1.0 - dones)
        * max_next_q
    )

    # --------------------------------------------------------
    # Huber loss
    # --------------------------------------------------------

    errors = chosen_q - targets

    loss_values, dq_selected = \
        huber_loss_and_gradient(errors)

    loss = np.mean(loss_values)

    # --------------------------------------------------------
    # Gradient of Q output
    # --------------------------------------------------------

    dq = np.zeros_like(q_values)

    dq[
        batch_indices,
        actions,
    ] = dq_selected / BATCH_SIZE

    # --------------------------------------------------------
    # Backpropagation
    # --------------------------------------------------------

    x, z1, a1, z2, a2 = cache

    dW3 = a2.T @ dq
    db3 = np.sum(
        dq,
        axis=0,
    )

    da2 = dq @ online.W3.T
    dz2 = da2 * (z2 > 0)

    dW2 = a1.T @ dz2
    db2 = np.sum(
        dz2,
        axis=0,
    )

    da1 = dz2 @ online.W2.T
    dz1 = da1 * (z1 > 0)

    dW1 = x.T @ dz1
    db1 = np.sum(
        dz1,
        axis=0,
    )

    gradients = [
        dW1,
        db1,
        dW2,
        db2,
        dW3,
        db3,
    ]

    # --------------------------------------------------------
    # Global gradient clipping
    # --------------------------------------------------------

    total_norm = 0.0

    for g in gradients:
        total_norm += np.sum(g * g)

    total_norm = np.sqrt(total_norm)

    if total_norm > MAX_GRAD_NORM:

        scale = MAX_GRAD_NORM / (
            total_norm + 1e-8
        )

        for g in gradients:
            g *= scale

    optimizer.step(gradients)

    return float(loss)


# ============================================================
# Epsilon schedule
# ============================================================

def epsilon_by_step(step):

    fraction = min(
        step / EPSILON_DECAY_STEPS,
        1.0,
    )

    return (
        EPSILON_START
        + fraction
        * (
            EPSILON_END
            - EPSILON_START
        )
    )


# ============================================================
# Action selection
# ============================================================

def select_action(
    network,
    state,
    epsilon,
):

    if rng.random() < epsilon:
        return int(
            rng.integers(
                network.num_actions
            )
        )

    state_batch = state[None, ...]

    q_values = network.predict(
        state_batch
    )[0]

    return int(np.argmax(q_values))


# ============================================================
# Training
# ============================================================

def train():

    env = PongEnv()

    print(
        "Actions:",
        env.num_actions,
    )

    print(
        "Observation:",
        (
            FRAME_STACK,
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
    )

    online = DQN(env.num_actions)
    target = DQN(env.num_actions)

    target.copy_from(online)

    optimizer = Adam(
        online,
        learning_rate=LEARNING_RATE,
    )

    replay = ReplayBuffer(
        REPLAY_SIZE
    )

    state, _ = env.reset(
        seed=SEED
    )

    episode_reward = 0.0
    episode = 0

    losses = []

    for step in range(1, MAX_STEPS + 1):

        epsilon = epsilon_by_step(
            step
        )

        # ----------------------------------------------------
        # Select action
        # ----------------------------------------------------

        action = select_action(
            online,
            state,
            epsilon,
        )

        # ----------------------------------------------------
        # Environment step
        # ----------------------------------------------------

        (
            next_state,
            reward,
            terminated,
            truncated,
            _,
        ) = env.step(action)

        done = (
            terminated
            or truncated
        )

        # ----------------------------------------------------
        # Store transition
        # ----------------------------------------------------

        replay.add(
            state,
            action,
            reward,
            next_state,
            float(done),
        )

        state = next_state

        episode_reward += reward

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        if (
            step >= LEARNING_STARTS
            and step % TRAIN_EVERY == 0
        ):

            loss = train_step(
                online,
                target,
                optimizer,
                replay,
            )

            losses.append(loss)

        # ----------------------------------------------------
        # Target network
        # ----------------------------------------------------

        if (
            step % TARGET_UPDATE_EVERY
            == 0
        ):

            target.copy_from(
                online
            )

        # ----------------------------------------------------
        # Episode finished
        # ----------------------------------------------------

        if done:

            mean_loss = (
                np.mean(losses[-100:])
                if losses
                else 0.0
            )

            print(
                f"step={step:8d} "
                f"episode={episode:5d} "
                f"reward={episode_reward:7.2f} "
                f"epsilon={epsilon:.3f} "
                f"loss={mean_loss:.5f} "
                f"replay={len(replay)}"
            )

            episode += 1
            episode_reward = 0.0

            state, _ = env.reset()

        # ----------------------------------------------------
        # Periodic checkpoint
        # ----------------------------------------------------

        if (
            step % 100_000
            == 0
        ):

            online.save(
                MODEL_FILE
            )

            print(
                f"Saved {MODEL_FILE}"
            )

    online.save(
        MODEL_FILE
    )

    env.close()


# ============================================================
# Evaluation
# ============================================================

def evaluate(
    model_file=MODEL_FILE,
    episodes=5,
):

    env = gym.make(
        ENV_ID,
        frameskip=1,
        repeat_action_probability=0.0,
        full_action_space=False,
        render_mode="human",
    )

    network = DQN(
        env.action_space.n
    )

    network.load(
        model_file
    )

    frame_stack = FrameStack(
        FRAME_STACK
    )

    for episode in range(episodes):

        obs, _ = env.reset()

        frame = preprocess_frame(
            obs
        )

        state = frame_stack.reset(
            frame
        )

        total_reward = 0.0

        while True:

            q = network.predict(
                state[None, ...]
            )[0]

            action = int(
                np.argmax(q)
            )

            total_reward_this_step = 0.0

            for _ in range(FRAME_SKIP):

                (
                    obs,
                    reward,
                    terminated,
                    truncated,
                    _,
                ) = env.step(action)

                total_reward_this_step += reward

                if (
                    terminated
                    or truncated
                ):
                    break

            frame = preprocess_frame(
                obs
            )

            state = frame_stack.append(
                frame
            )

            total_reward += (
                total_reward_this_step
            )

            if (
                terminated
                or truncated
            ):
                break

        print(
            f"Evaluation episode "
            f"{episode + 1}: "
            f"{total_reward:.1f}"
        )

    env.close()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    train()

    # After training, uncomment this:
    #
    # evaluate()