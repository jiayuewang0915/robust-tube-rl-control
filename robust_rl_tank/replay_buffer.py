from collections import deque, namedtuple

import numpy as np


Experience = namedtuple("Experience", ["state", "error", "action", "reward", "next_state", "next_error", "done", "ID"])
ExperienceModel = namedtuple("ExperienceModel", ["state", "error", "action", "reward", "next_state", "next_error", "done"])
ExperienceError = namedtuple(
    "ExperienceError",
    ["state", "error", "state_er", "action_er", "reward_error", "next_state", "next_error", "next_state_er", "done"],
)


class ExperienceReplayBuffer:
    """Replay buffer variant for paired experiences from two model IDs."""

    def __init__(self, maximum_length):
        self.buffer = deque(maxlen=maximum_length)

    def append(self, experience):
        self.buffer.append(experience)

    def __len__(self):
        return len(self.buffer)

    def sample_batch(self, n, to_append=None):
        if n > len(self.buffer):
            raise IndexError("Tried to sample too many elements from the buffer.")

        indices1 = [index for index, value in enumerate(self.buffer) if value[7] == 1]
        indices2 = [index for index, value in enumerate(self.buffer) if value[7] == 2]
        indices1 = np.random.choice(indices1, size=n, replace=False)
        indices2 = np.random.choice(indices2, size=n, replace=False)

        batch1 = [self.buffer[i] for i in indices1]
        batch2 = [self.buffer[i] for i in indices2]
        if to_append is not None:
            batch1.append(to_append)
            batch2.append(to_append)
        return zip(*batch1), zip(*batch2)


class ExperienceReplayBufferError:
    """Uniform replay buffer used by both current DDPG agents."""

    def __init__(self, maximum_length):
        self.buffer = deque(maxlen=maximum_length)

    def append(self, experience):
        self.buffer.append(experience)

    def __len__(self):
        return len(self.buffer)

    def sample_batch(self, n, to_append=None):
        if n > len(self.buffer):
            raise IndexError("Tried to sample too many elements from the buffer.")

        # Sampling without replacement reduces duplicated transitions inside a
        # mini-batch while keeping the implementation deterministic in shape.
        indices = np.random.choice(len(self.buffer), size=n, replace=False)
        batch = [self.buffer[i] for i in indices]
        if to_append is not None:
            batch.append(to_append)
        return zip(*batch)


class ExperienceReplayBufferModel(ExperienceReplayBufferError):
    """Alias class for readability: model and error agents store different tuples."""

    pass


# Legacy-style aliases are kept for users who compare this package with older
# experiment notes, while the clean class names above are used by the package.
Experience_model = ExperienceModel
Experience_error = ExperienceError
ExperienceReplayBuffer_model = ExperienceReplayBufferModel
ExperienceReplayBuffer_error = ExperienceReplayBufferError
