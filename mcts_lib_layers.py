import math
import queue
import go
import numpy as np

# Set a few global variables.
PUCT_CONST = 0.03
INITIAL_TEMP = 1
FINAL_TEMP = 0.01
TEMP_SWITCH_THRESHOLD = 30


class MCTSNode:
    """ Class representing a node of the monte carlo search tree. """

    def __init__(self, state, parent=None, action_idx=None):
        """
        :param state: a 17x13x13 array representing board state (self.expand() will add two layers for liberty counts).
        """
        self.state = state

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
        """
        The move that is finally selected is the move with the most visits, relative to the total number of visits made,
        with adjustments made by a temperature. See page 8 of: https://www.nature.com/articles/nature24270.epdf
        Ensure that only moves that have been expanded are selected (noise could select something else).
        :param temperature: either 1, or close to 0. If 1, more exploration. If ~0, the most visited move always picked.
        :return:
        """
        exponentiated_visits = [vis_cnt ** (1.0 / temperature) for vis_cnt in self.visit_count]
        total_exponentiated_visits = sum(exponentiated_visits)
        dirchlect_noise_coeff = 3.0 * sum(self.visit_count) / 100.0  # 3% of the total exponentiated visit count.
        dirichlet_probs = np.random.dirichlet([dirchlect_noise_coeff]*170)
        max_idx = 169
        max_value = (dirichlet_probs[max_idx] + exponentiated_visits[max_idx]) / total_exponentiated_visits
        for idx in range(170):
            if idx not in self.children:
                continue
            curr_value = (dirichlet_probs[idx] + exponentiated_visits[idx]) / total_exponentiated_visits
            if curr_value >= max_value:
                max_value = curr_value
                max_idx = idx
        return max_idx

    def back_propagate(self, sim_result):
        """
        Now that this node has been expanded, share its newly obtained insights with its ancestral nodes.
        :param sim_result: the child's (this nodes) win likelihood after the parent plays at this node's action index.
        """
        # If this node is the root, it does not have any parents to update.
        if not self.parent:
            return

        # Otherwise, let the parent node know that we visited this index.
        self.parent.increment_visit_count_at_idx(self.action_idx)

        # Use a negated sim result for backing up the parent, since the parent is on the other team.
        negated_sim_result = -sim_result
        self.parent.update_total_value_estimate_at_idx(self.action_idx, negated_sim_result)
        self.parent.update_mean_value_estimate_at_idx(self.action_idx)

        # Now the parent node needs to update its parents.
        self.parent.back_propagate(negated_sim_result)

    def increment_visit_count_at_idx(self, idx):
        """ Update the count for the number of times the child node at the given index has been visited. """
        self.visit_count[idx] += 1

    def update_total_value_estimate_at_idx(self, idx, value):
        """ Increment the total estimated value for playing at this index. """
        self.total_action_value[idx] += value

    def update_mean_value_estimate_at_idx(self, idx):
        """ Determine the mean estimated value for playing at this index. """
        self.mean_action_value[idx] = float(self.total_action_value[idx]) / float(self.visit_count[idx])

    def expand(self, processing_queue):
        """
        Expand this node by placing its board state on the policy value network's processing queue & getting the result.
        :param processing_queue: a queue managed by the main self play thread, for evaluating board states.
        :return: the predicted value for selecting the best move, given the node's board state.
        """

        # First, get the liberty layers, given the current state.
        self.state = np.reshape(self.state, (17, 13, 13))
        current_self_stones = self.state[14]
        current_opp_stones = self.state[15]
        if self.state[16][0][0] == 1:  # If the player to make the move is black, the self value is 1 and the opp val is 2.
            friendly_val = 1
            enemy_val = 2
        else:
            friendly_val = 2
            enemy_val = 1
        curr_board = current_self_stones*friendly_val + current_opp_stones*enemy_val
        friendly, enemy = go.get_liberty_counts_from_board(curr_board, friendly_val)
        friendly = np.reshape(friendly, (13, 13))
        enemy = np.reshape(enemy, (13, 13))

        # Append the liberty layers to the state.
        state_w_libs = [x for x in self.state]
        state_w_libs.append(friendly)
        state_w_libs.append(enemy)

        # Predict a policy and value, by sending the state to the processing thread. Wait for a response.
        response_queue = queue.Queue()
        transmission = {"state": state_w_libs, "response_queue": response_queue}
        processing_queue.put(transmission)
        reception = response_queue.get()  # This is a blocking get (thread will wait for response before continuing).
        predicted_policy = reception["policy"]  # A 170 value array of weighted move options.
        predicted_value = reception["value"]  # A value indicating likelihood of winning.

        # Filter out illegal moves.
        legal_moves = self.get_legal_actions()
        best_legal_moves = legal_moves*predicted_policy

        # Set the nodes main attributes, using correctly shaped np arrays.
        self.predicted_value = np.reshape(predicted_value, 1)           # V(s)
        self.prior_probs = np.reshape(best_legal_moves, 170)            # P(s, a)
        self.legal_actions = legal_moves
        self.visit_count = np.zeros(170)                                # N(s, a)
        self.total_action_value = np.zeros(170)                         # W(s, a)
        self.mean_action_value = np.zeros(170)                          # Q(s, a)
        self.fully_expanded = True
        return self.predicted_value

    def get_legal_actions(self, pass_illegal=True):
        """
        Identify the legal moves, given the current board state.
        :param pass_illegal: should the tree search be allowed to suggest passing?
        :return: an 170 item array containing the legal moves.
        """
        legal_moves = go.get_all_legal_moves_from_board_state(self.state)
        legal_moves.append(0) if pass_illegal else legal_moves.append(1)
        legal_moves = np.asarray(legal_moves)
        return legal_moves

    def apply_dirichlet_noise(self, prior_probs):
        dirichlet_probs = np.random.dirichlet([0.03]*170)
        new_prior_probs = [0.75*prior_probs[idx] + 0.25*dirichlet_probs[idx] for idx in range(170)]
        return new_prior_probs

    def select_child_with_best_puct(self):
        """
        During traversal (through nodes that have already been expanded) we must select a non-expanded child node
        with a maximum Polynomial Upper Confidence for Trees (PUCT) score, adjusted with dirichlet noise.
        :return: a node that has not yet been expanded.
        """
        # If the node is root, apply Dirichlet noise to the prior probs.
        # We cant update the prior probs directly because we call this function for every traversal (simulation).
        adjusted_prior_probs = self.prior_probs
        if self.parent is None:
            adjusted_prior_probs = self.apply_dirichlet_noise(self.prior_probs)
        max_action = 169
        total_visits = sum(self.visit_count)
        max_value = self.legal_actions[max_action] * (self.mean_action_value[max_action] + PUCT_CONST * adjusted_prior_probs[max_action] * math.sqrt(total_visits) / (1.0 + self.visit_count[max_action]))
        for idx in range(169):
            if self.legal_actions[idx] != 1:
                continue
            puct_value = self.mean_action_value[idx] + PUCT_CONST * adjusted_prior_probs[idx] * math.sqrt(total_visits) / (1.0 + self.visit_count[idx])
            if puct_value >= max_value:
                max_value = puct_value
                max_action = idx

        # Given the move w/ the best predicted value, create a child (if DNE) to represent the board given this action.
        if max_action not in self.children:

            # Update the board state (making a copy) & represent the board from the child's (next player's) perspective.
            child_board_state = go.update_board_state_for_move(max_action, self.state)

            # Create the child node and add it to this nodes list of children.
            child_node = MCTSNode(child_board_state, parent=self, action_idx=max_action)
            self.children[max_action] = child_node

        # Return the child node identified by the move with the best PUCT value.
        return self.children[max_action]

    def send_parent_to_nursing_home(self):
        """ Perform garbage collection on the node's parent. Do this if the node is now the new root."""
        self.parent = None

    def get_board_state(self):
        """ Provide the current board state of this node. """
        return self.state


class MonteCarloSearchTree:
    """
    This class manages the tree search process.
    """

    def __init__(self, tree_id, state_processing_queue, root_state, root=None):
        """
        This class handles the tree search process.
        :param tree_id: the identify of this tree, as a unique integer.
        :param state_processing_queue: a queue of states which need a policy / value evaluation.
        :param root_state: the current state of the game, represented as a 17x13x13 array.
        :param root: a MCTS node, containing a history of evaluated moves.
        """
        self.tree_id = tree_id
        self.state_processing_queue = state_processing_queue
        self.root = root if root else MCTSNode(root_state)  # Create the root node, if it hasn't been provided.
        self.root.send_parent_to_nursing_home()  # Make sure the root does not have parents. Brutal...
        self.temperature = INITIAL_TEMP
        self.search_count = 0

    def update_root(self, new_root_node):
        """
        The tree will be reused during self play, so the root must be updated to represent the new board state.
        :param new_root_node: the child of the current root node which represents the move selected during MCTS.
        """
        self.root = new_root_node
        self.root.send_parent_to_nursing_home()  # Since this is the new root, it shouldn't have parents.

    def search(self, num_simulations):
        """
        Select the best move by expanding the search tree.
        Set the new root of the tree as the state at that index (as an MCTS node).
        :param num_simulations: the total number of potential future moves to examine.
        :return: the index of the top move
        """
        # If this tree has reached the temp switch threshold, update it.
        if self.search_count > TEMP_SWITCH_THRESHOLD:
            self.temperature = FINAL_TEMP
        self.search_count += 1

        # Conduct a ton of simulations.
        for _ in range(num_simulations):
            leaf = self.traverse(self.root)  # leaf = unvisited node
            sim_result = leaf.expand(self.state_processing_queue)
            leaf.back_propagate(sim_result)

        # Given the values obtained through simulation, select the best action.
        best_action_idx = self.root.select_best_action(self.temperature)

        # In order to reuse the tree, update the root as the child who represent the selected move.
        self.update_root(self.root.children[best_action_idx])

        # Return the index of the best move identified MCTS.
        return best_action_idx

    @staticmethod
    def traverse(node):
        """
        Walk the tree (typically starting at the root node) until we reach a node that has not been expanded.
        :param node:
        :return:
        """
        while node.fully_expanded:
            node = node.select_child_with_best_puct()
        return node


