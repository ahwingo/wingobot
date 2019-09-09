from go import *
import math
import numpy as np

PUCT_CONST = 0.03

class MCTSNode:

    def __init__(self, state, parent=None, action_idx=None):
        """
        :param state: a 13x13x17 ndarray representing board state
        """
        self.state = state
        previous_self_stones = self.state[12]
        previous_opp_stones = self.state[13]
        current_self_stones = self.state[14]
        current_opp_stones = self.state[15]
        self.previous_board = [previous_self_stones[idx] - previous_opp_stones[idx] for idx in range(169)]
        self.current_board = [current_self_stones[idx] - current_opp_stones[idx] for idx in range(169)]
        self.fully_expanded = False
        self.parent = parent

        self.action_idx = action_idx

        self.predicted_value = None             # V(s)
        self.prior_probs = None                 # P(s, a)
        self.visit_count = None                 # N(s, a)
        self.total_action_value = None          # W(s, a)
        self.mean_action_value = None           # Q(s, a)
        self.legal_actions = None               # Binary matrix of which actions are legal.

        # A list of all children (at next legal states) that have been turned into nodes.
        self.children = {}

    def select_best_action(self, temperature):
        total_visits = sum(self.visit_count)
        max_idx = 169
        max_value = (self.visit_count[max_idx] ** (1.0 / temperature)) / (total_visits ** (1.0 / temperature))
        for idx in range(170):
            curr_value = (self.visit_count[idx] ** (1.0 / temperature)) / (total_visits ** (1.0 / temperature))
            if curr_value >= max_value:
                max_value = curr_value
                max_idx = idx
        return max_idx

    def backpropagate(self, sim_result):
        if not self.parent:
            return
        self.parent.increment_visit_count_at_idx(self.action_idx)
        # Use a negated sim result for backing up the parent, since the parent is on the other team.
        negated_sim_result = -sim_result
        self.parent.update_total_value_estimate_at_idx(self.action_idx, negated_sim_result)
        self.parent.update_mean_value_estimate_at_idx(self.action_idx)
        self.parent.backpropagate(negated_sim_result)

    def increment_visit_count_at_idx(self, idx):
        self.visit_count[idx] += 1

    def update_total_value_estimate_at_idx(self, idx, value):
        self.total_action_value[idx] += value

    def update_mean_value_estimate_at_idx(self, idx):
        self.mean_action_value[idx] = float(self.total_action_value[idx]) / float(self.visit_count[idx])

    def expand(self, evaluator_network):
        prior_probs, predicted_value = evaluator_network.predict_given_state(np.reshape(self.state, (1, 13, 13, 17)))
        self.legal_actions = self.get_legal_actions()
        self.predicted_value = np.reshape(predicted_value, 1)         # V(s)
        self.prior_probs = np.reshape(prior_probs, 170)               # P(s, a)
        self.visit_count = [0]*170                                      # N(s, a)
        self.total_action_value = [0]*170                               # W(s, a)
        self.mean_action_value = [0]*170                                # Q(s, a)
        self.fully_expanded = True
        return predicted_value

    def get_legal_actions(self):
        legal_actions = [0]*169
        for idx in range(169):
            if board_pos_after_move(self.current_board.copy(), self.previous_board.copy(), idx, 1)["move_legal"]:
                legal_actions[idx] = 1
        legal_actions.append(1)  # The last move is a pass and that is always legal!!!
        return legal_actions

    def apply_dirichlet_noise(self, prior_probs):
        dirichlet_probs = np.random.dirichelt(prior_probs, 0.03)
        new_prior_probs = [0.75*prior_probs[idx] + 0.25*dirichlet_probs[idx] for idx in range(170)]
        return new_prior_probs

    def select_child_with_best_puct(self):
        # If the node is root, apply Dirichlet noise to the prior probs.
        if self.parent is None:
            self.prior_probs = self.apply_dirichlet_noise(self.prior_probs)
    @staticmethod
    def traverse(node):
        while node.fully_expanded:
            node = node.select_child_with_best_puct()
        return node


from go import *
import math
import numpy as np

PUCT_CONST = 0.03

class MCTSNode:

    def __init__(self, state, parent=None, action_idx=None):
        """
        :param state: a 13x13x17 ndarray representing board state
        """
        self.state = state
        previous_self_stones = self.state[12]
        previous_opp_stones = self.state[13]
        current_self_stones = self.state[14]
        current_opp_stones = self.state[15]
        self.previous_board = [previous_self_stones[idx] - previous_opp_stones[idx] for idx in range(169)]
        self.current_board = [current_self_stones[idx] - current_opp_stones[idx] for idx in range(169)]
        self.fully_expanded = False
        self.parent = parent

        self.action_idx = action_idx

        self.predicted_value = None             # V(s)
        self.prior_probs = None                 # P(s, a)
        self.visit_count = None                 # N(s, a)
        self.total_action_value = None          # W(s, a)
        self.mean_action_value = None           # Q(s, a)
        self.legal_actions = None               # Binary matrix of which actions are legal.

        # A list of all children (at next legal states) that have been turned into nodes.
        self.children = {}

    def select_best_action(self, temperature):
        total_visits = sum(self.visit_count)
        max_idx = 169
        max_value = (self.visit_count[max_idx] ** (1.0 / temperature)) / (total_visits ** (1.0 / temperature))
        for idx in range(170):
            curr_value = (self.visit_count[idx] ** (1.0 / temperature)) / (total_visits ** (1.0 / temperature))
            if curr_value >= max_value:
                max_value = curr_value
                max_idx = idx
        return max_idx

    def backpropagate(self, sim_result):
        if not self.parent:
            return
        self.parent.increment_visit_count_at_idx(self.action_idx)
        # Use a negated sim result for backing up the parent, since the parent is on the other team.
        negated_sim_result = -sim_result
        self.parent.update_total_value_estimate_at_idx(self.action_idx, negated_sim_result)
        self.parent.update_mean_value_estimate_at_idx(self.action_idx)
        self.parent.backpropagate(negated_sim_result)

    def increment_visit_count_at_idx(self, idx):
        self.visit_count[idx] += 1

    def update_total_value_estimate_at_idx(self, idx, value):
        self.total_action_value[idx] += value

    def update_mean_value_estimate_at_idx(self, idx):
        self.mean_action_value[idx] = float(self.total_action_value[idx]) / float(self.visit_count[idx])

    def expand(self, evaluator_network):
        prior_probs, predicted_value = evaluator_network.predict_given_state(np.reshape(self.state, (1, 13, 13, 17)))
        self.legal_actions = self.get_legal_actions()
        self.predicted_value = np.reshape(predicted_value, 1)         # V(s)
        self.prior_probs = np.reshape(prior_probs, 170)               # P(s, a)
        self.visit_count = [0]*170                                      # N(s, a)
        self.total_action_value = [0]*170                               # W(s, a)
        self.mean_action_value = [0]*170                                # Q(s, a)
        self.fully_expanded = True
        return predicted_value

    @staticmethod
    def traverse(node):
        while node.fully_expanded:
            node = node.select_child_with_best_puct()
        return node


