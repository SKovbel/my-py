import copy
import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import BoundedArraySpec
from tf_agents.trajectories.time_step import StepType
from tf_agents.trajectories.time_step import TimeStep


class TicTacToeEnvironment(py_environment.PyEnvironment):
  REWARD_WIN = np.asarray(1.0, dtype=np.float32)
  REWARD_LOSS = np.asarray(-1.0, dtype=np.float32)
  REWARD_DRAW_OR_NOT_FINAL = np.asarray(0.0, dtype=np.float32)
  REWARD_ILLEGAL_MOVE = np.asarray(-0.001, dtype=np.float32)

  REWARD_WIN.setflags(write=False)
  REWARD_LOSS.setflags(write=False)
  REWARD_DRAW_OR_NOT_FINAL.setflags(write=False)

  def __init__(self, rng: np.random.RandomState = None, discount=1.0):
    super(TicTacToeEnvironment, self).__init__(handle_auto_reset=True)
    self._rng = rng
    self._discount = np.asarray(discount, dtype=np.float32)
    self._states = None

  def action_spec(self):
    return BoundedArraySpec((2,), np.int32, minimum=0, maximum=2)

  def observation_spec(self):
    return BoundedArraySpec((3, 3), np.int32, minimum=0, maximum=2)

  def _reset(self):
    self._states = np.zeros((3, 3), np.int32)
    return TimeStep(
        StepType.FIRST,
        np.asarray(0.0, dtype=np.float32),
        self._discount,
        self._states,
    )

  def _legal_actions(self, states: np.ndarray):
    return list(zip(*np.where(states == 0)))

  def _opponent_play(self, states: np.ndarray):
    actions = self._legal_actions(np.array(states))
    if not actions:
      raise RuntimeError('There is no empty space for opponent to play at.')

    if self._rng:
      i = self._rng.randint(len(actions))
    else:
      i = 0
    return actions[i]

  def get_state(self) -> TimeStep:
    # Returning an unmodifiable copy of the state.
    return copy.deepcopy(self._current_time_step)

  def set_state(self, time_step: TimeStep):
    self._current_time_step = time_step
    self._states = time_step.observation

  def _step(self, action: np.ndarray):
    action = tuple(action)
    if self._states[action] != 0:
      return TimeStep(
          StepType.LAST,
          TicTacToeEnvironment.REWARD_ILLEGAL_MOVE,
          self._discount,
          self._states,
      )

    self._states[action] = 1

    is_final, reward = self._check_states(self._states)
    if is_final:
      return TimeStep(StepType.LAST, reward, self._discount, self._states)

    # TODO(b/152638947): handle multiple agents properly.
    # Opponent places '2' on the board.
    opponent_action = self._opponent_play(self._states)
    self._states[opponent_action] = 2

    is_final, reward = self._check_states(self._states)

    step_type = StepType.MID
    if np.all(self._states == 0):
      step_type = StepType.FIRST
    elif is_final:
      step_type = StepType.LAST

    return TimeStep(step_type, reward, self._discount, self._states)

  def _check_states(self, states: np.ndarray):
    seqs = np.array([
        # each row
        states[0, :],
        states[1, :],
        states[2, :],
        # each column
        states[:, 0],
        states[:, 1],
        states[:, 2],
        # diagonal
        states[(0, 1, 2), (0, 1, 2)],
        states[(2, 1, 0), (0, 1, 2)],
    ])
    seqs = seqs.tolist()
    if [1, 1, 1] in seqs:
      return True, TicTacToeEnvironment.REWARD_WIN  # win
    if [2, 2, 2] in seqs:
      return True, TicTacToeEnvironment.REWARD_LOSS  # loss
    if 0 in states:
      # Not final
      return False, TicTacToeEnvironment.REWARD_DRAW_OR_NOT_FINAL
    return True, TicTacToeEnvironment.REWARD_DRAW_OR_NOT_FINAL  # draw