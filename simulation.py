import numpy as np
# import matplotlib.pyplot as plt
from scipy.spatial import distance
import time
import copy
import sys

def closest(a, b, xy):
    """ Calculates closest point on line between and b given a pod at location xy
        args:
            a [int,int]: starting point of line
            b [int,int]: ending point of line
            xy [int,int]: current location
    """
    da = b[1] - a[1]
    db = a[0] - b[0]
    c1 = da*a[0] + db*a[1]
    c2 = -db*xy[0] + da*xy[1]
    det = da*da + db*db

    if det:
        cx = (da*c1 - db*c2) / det
        cy = (da*c2 + db*c1) / det
    else:
        cx, cy = xy

    return np.array([cx, cy])

def collision(xy1, xy2, vxy1, vxy2):
    """ Calculate collision point between two moving objects.
        Returns (distance to collision) / (distance moved if no collision) or None if no collision occurs
        args:
            xy1 [int,int]: position of pod 1
            xy2 [int,int]: position of pod 2
            vxy1 [int,int]: velocity of pod 1
            vxy2 [int,int]: velocity of pod 2
    """
    dist2 = sum((xy1 - xy2)**2)
    sr = 640000
    if dist2 < sr:
        return 0.

    if all(vxy1 == vxy2):
        return None

    xy = xy1 - xy2
    vxy = vxy1 - vxy2

    p = closest(xy, xy + vxy, (0,0))

    pdist = sum(p**2)
    mypdist = sum((p-xy)**2)

    if (pdist < sr):
        length = np.sqrt(sum(vxy**2))
        backdist = np.sqrt(sr - pdist)
        p = p - backdist * vxy / length
        newdist = sum((xy - p)**2)

        if newdist > mypdist:
            return None

        pdist = np.sqrt(newdist)

        if (pdist > length):
            return None

        return pdist / length
    
    return None

def get_angle(xy1, xy2):
    """ Calculates the angle between xy1 and xy2 in radians.
        args:
            xy1 [int,int]: point 1
            xy2 [int,int]: point 2
    """
    d = np.sqrt(np.sum((xy1 - xy2)**2))
    dxy = (xy2 - xy1) / d
    a = np.arccos(dxy[0])
    if dxy[1] < 0:
        a = TWO_PI - a

    return a;


def diff_angle(xy1, xy2, angle):
    """ Calculates required change in angle for pod at xy1 to get to xy2
        args:
            xy1 [int,int]: Current location
            xy2 [int,int]: Desired location
            angle [float]: Current angle in radians
    """
    a = get_angle(xy1, xy2);

    right = a - angle if angle <= a else TWO_PI - angle + a
    left = angle - a if angle >= a else angle + TWO_PI - a
    da = -left if left <= right else right
    
    return da


PI = np.pi
TWO_PI = 2* np.pi

class Pods:
    def __init__(self, checkpoints, req_laps):
        """ Initialize class Pods
            Pods represents all 4 pods in a game i.e. has 4 of each value
            args:
                checkpoints [list of (int, int)]: checkpoints on the map. checkpoint 0 is starting location
                req_laps [int]: number of laps required to win race
        """
        self.xy = np.zeros((4, 2)), np.zeros((4, 2))
        self.vxy = np.zeros((4, 2)), np.zeros((4, 2))
        self.angle = np.zeros((4))
        self.shields = np.zeros(4)
        self.self_hit = 0
        self.shield_hit = np.zeros(4)
        # self.boost_used = np.array([True] * 4)
        self.checkpoints = np.array(checkpoints)
        self.cp_id = np.zeros(4, dtype=int)
        self.num_cp = len(checkpoints)
        # self.previous_best = None
        self.lap = np.zeros(4)
        self.next_cp = np.zeros((4, 2))
        self.cp_passed = np.zeros(4)
        self.req_laps = req_laps

    def update_pod(self, x, y, vx, vy, angle, cp_id):
        """ Update pods' positions, velocities, angles, and checkpoints based on information from game
            args:
                x, y [list of ints]: cartesian position of pods
                vx, vy [list of ints]: cartesian velocities of pods
                angle [list of ints]: angles (in degrees) of pods
                cp_id [list of ints]: id of next checkpoint for each pod
        """
        self.xy = np.concatenate((np.expand_dims(np.array(x, dtype=float), axis=1), np.expand_dims(np.array(y, dtype=float), axis=1)), axis=1)
        self.vxy = np.concatenate((np.expand_dims(np.array(vx, dtype=float), axis=1), np.expand_dims(np.array(vy, dtype=float), axis=1)), axis=1)
        self.angle = np.array(angle, dtype=float)*PI/180. # convert to radians
        cp_id = np.array(cp_id)
        if any(cp_id != self.cp_id % self.num_cp):
            self.cp_passed[cp_id != self.cp_id] += 1
            self.cp_id = cp_id % self.num_cp
            self.lap[cp_id == 1] += 1 
        self.next_cp = self.checkpoints[self.cp_id]

    def dist_to_cp(self):
        """ Calculate squared distance of each pod to their respective checkpoints """
        a = (self.xy - self.next_cp)**2 # + (self.y - self.next_cp[:, 1])**2
        return a[:, 0] + a[:, 1]

    def bounce(self, i, j):
        """ Update velocities due to a collision
            args:
                i [int]: index of a collidee
                j [int]: index of the other collidee
        """
        dxy = self.xy[i, :] - self.xy[j, :]
        dv = self.vxy[i, :] - self.vxy[j, :]
        m1, m2 = (self.shields[i] == 3) * 9. + 1., (self.shields[j] == 3) * 9. + 1.
        mcoeff = (m1 + m2)/(m1 * m2)
        prod = sum(dxy * dv)
        fxy = dxy * prod / (sum((self.xy[i]-self.xy[j])**2) * mcoeff)
        self.vxy[i] -= fxy / m1
        self.vxy[j] -= fxy / m2

        imp = sum(fxy**2)
        if imp < 14400:
            fxy = fxy * np.sqrt(14400./imp)

        self.vxy[i] -= fxy / m1
        self.vxy[j] -= fxy / m2

    def rotate(self, angle):
        """ Rotate pod by given angle
            args:
                angle [float]: angle to rotate by
        """
        self.angle = (self.angle + angle) % TWO_PI

    def accelerate(self, thrust):
        """ Update velocities and shields given thrust. 
            A pod cannot accelerate for 3 turns after using a shield
            args:
                thrust [int]: acceleration
        """
        self.shields[(thrust == -1)] = 3

        can_acc = (self.shields == 0)
        self.vxy[can_acc, 0] += np.cos(self.angle[can_acc]) * thrust[can_acc]
        self.vxy[can_acc, 1] += np.sin(self.angle[can_acc]) * thrust[can_acc]

    def move(self):
        """ Given velocities and angles move pods
            Will account for up to 1 collision.
        """
        # Check to see if there are any collisions between pods
        collisions = [(i, j, collision(self.xy[i], self.xy[j], self.vxy[i], self.vxy[j])) for i in xrange(4) for j in xrange(4) if i != j]
        c = min(collisions, key=lambda x: x[2] or 2.)

        if c[2] is None:
            # If there are no collisions, move pods full distance
            self.xy += self.vxy * 1.0
        else:
            # If there is a collision, move pods up to time of collision
            self.xy += self.vxy * c[2]
            # Update collision counters used for fitness functions
            if c[0] + c[1] == 1:
                self.self_hit += abs(sum(self.vxy[0] - self.vxy[1]))
            elif self.shields[c[0]]:
                self.shield_hit[c[0]] += abs(sum(self.vxy[0] - self.vxy[1]))
            elif self.shields[c[1]]:
                self.shield_hit[c[1]] += abs(sum(self.vxy[0] - self.vxy[1]))
            # Update velocities due to collision  
            self.bounce(c[0], c[1])
            # Move pods for remaining time in turn
            self.xy += self.vxy * (1. - c[2])


    def end(self):
        """ Update required parameters to prepare for the next iteration """
        # To be more accurate with game
        self.xy = np.round(self.xy)
        self.vxy = (self.vxy * 0.85).astype(int).astype(float)
        # Update shield thrust prevention turn count
        self.shields[(self.shields > 0)] -= 1

        # Update checkpoints if checkpoints are reached
        # reduced 360000 (600^2) to 2500000 (500^2) to ensure that simulation errors don't make me miss checkpoints
        reached_cp = self.dist_to_cp() < 250000
        if any(reached_cp):
            self.cp_id[reached_cp] = (self.cp_id[reached_cp] + 1) % self.num_cp
            for i in xrange(4):
                if reached_cp[i] and self.cp_id[i] == 1:
                    self.lap[i] += 1 
            self.cp_passed[reached_cp] += 1
            self.next_cp[reached_cp] = self.checkpoints[self.cp_id[reached_cp]]

    def play(self, move):
        """ Play a single iteration (1 move) of a game
            args:
                move [np array]: move to play (contains a move for all 4 pods)
        """
        self.rotate(move[:, 0])
        self.accelerate(move[:, 1])
        self.move()
        self.end()

    def evaluate_pos(self):
        """ Proportional to true score of pod i.e. returns a higher score for pods that are ahead in a race """
        return self.cp_passed*5e10 - self.dist_to_cp()*2 #+ 1e20 * (self.lap == self.req_laps)

    def get_target(self, angle):
        """ Given a change in angle, find an x, y along the line of the new angle
            Used to convert from angle in my simulatin to target_x, target_y required for game input
            args:
                angle [float]: Desired angle change
        """
        target_angle = (self.angle + angle) % TWO_PI

        return self.xy[:, 0] + np.cos(target_angle) * 10000., self.xy[:, 1] + np.sin(target_angle) * 10000.


class Evolution:
    def __init__(self, pods, pop_size=4, num_turns=4):
        """ Initialize evolution class 
            args:
                pods [np array]: pods that will be uesd in evolution
                pop_size [int]: Population size to be used in evolutions
                num_turns [int]: number of turns to simulate
        """
        self.pods = pods
        self.start_state = None
        self.pop = None
        self.pop_size = pop_size
        self.num_turns = num_turns
        # Set previous best to straight forward and fast since we always start race facing first checkpoint
        self.my_prev_best = np.zeros([2, self.num_turns, 2])
        self.enemy_prev_best = np.zeros([2, self.num_turns, 2])
        self.my_prev_best[:,:,1] = 100
        self.enemy_prev_best[:,:,1] = 100

    def generate_population(self, my_sim):
        """ Generate starting population.
            See evolve method docstring for description of an invididual.
            Uses basic logic to find decent starting points for evolution.
            Only generates population size - 1 individuals because the remaining individual will be sourced from the best of the previous evolution
            args:
                my_sim [Bool]: True if evolving my pods, False if evolving enemy pods
        """
        # Create required shape
        pop = np.zeros([2, self.pop_size, self.num_turns, 2])
        # Determine runner (R) and bruiser (B) pods
        scores = self.pods.evaluate_pos()
        R = np.argmax(scores[:2]) if my_sim else np.argmax(scores[2:])
        enemy_R = np.argmax(scores[2:]) +2 if my_sim else np.argmax(scores[:2]) + 2
        B = int(not R)
        # runner prime (Rp) and bruiser prime (Bp) take into account who is being simulated
        Rp, Bp = (R, B) if my_sim else (R + 2, B + 2)
        # Runner distance to checkpoint used to factor thrust
        d_cp = 1.0 - (max(0., 3.e6 - sum((self.pods.xy[Rp] - self.pods.next_cp[Rp])**2)) / 3.e6)

        # Child 1
        # Bruiser aims for enemy runner, Runner aims for next next checkpoint
        # Angles
        B_diff_angles = diff_angle(self.pods.xy[Bp], self.pods.xy[enemy_R], self.pods.angle[Bp])
        B_max_num_turns = (np.abs(B_diff_angles) / 0.314159).astype(int)
        R_diff_angles = diff_angle(self.pods.xy[Rp], self.pods.checkpoints[(self.pods.cp_id[Rp] + 1) % self.pods.num_cp], self.pods.angle[Rp])
        R_max_num_turns = (np.abs(R_diff_angles) / 0.314159).astype(int)
        pop[R, 1, :R_max_num_turns + 1, 0] = R_diff_angles / (R_max_num_turns + 1) if R_max_num_turns + 1 <=4 else (0.314159 if R_diff_angles > 0 else -0.314159)
        pop[B, 1, :B_max_num_turns + 1, 0] = B_diff_angles / (B_max_num_turns + 1) if B_max_num_turns + 1 <=4 else (0.314159 if B_diff_angles > 0 else -0.314159)
        # Thrust
        pop[R, 1, :, 1] = 100 * d_cp
        pop[B, 1, :, 1] = 100

        # Child 2
        # Bruiser aims for enemy runner's next checkpoint, Runner aims for next checkpoint
        # Angles
        B_diff_angles = diff_angle(self.pods.xy[Bp], self.pods.next_cp[enemy_R], self.pods.angle[Bp])
        B_max_num_turns = (np.abs(B_diff_angles) / 0.314159).astype(int)
        R_diff_angles = diff_angle(self.pods.xy[Rp], self.pods.next_cp[Rp], self.pods.angle[Rp])
        R_max_num_turns = (np.abs(R_diff_angles) / 0.314159).astype(int)
        pop[R, 2, :R_max_num_turns + 1, 0] = R_diff_angles / (R_max_num_turns + 1) if R_max_num_turns + 1 <=4 else (0.314159 if R_diff_angles > 0 else -0.314159)
        pop[B, 2, :B_max_num_turns + 1, 0] = B_diff_angles / (B_max_num_turns + 1) if B_max_num_turns + 1 <=4 else (0.314159 if B_diff_angles > 0 else -0.314159)
        # Thrust
        pop[R, 2, :, 1] = 100 * d_cp
        pop[B, 2, :, 1] = 100

        # Child 3 
        # Bruiser aims for enemy runner's next next checkpoint
        # Runner hard turn with low speed
        # Angles
        B_diff_angles = diff_angle(self.pods.xy[Bp], self.pods.checkpoints[(self.pods.cp_id[enemy_R] + 1) % self.pods.num_cp], self.pods.angle[Bp])
        B_max_num_turns = (np.abs(B_diff_angles) / 0.314159).astype(int)
        pop[R, 3, :, 0] = 0.314159 if R_diff_angles > 0 else -0.314159
        pop[B, 3, :B_max_num_turns + 1, 0] = B_diff_angles / (B_max_num_turns + 1) if B_max_num_turns + 1 <=4 else (0.314159 if B_diff_angles > 0 else -0.314159)
        # Thrust
        pop[R, 3, :, 1] = 0
        pop[B, 3, :, 1] = 60
        
        return pop

    def breed(self, scores):
        """ Breed a child using the genes of the best runner and the genes of the best bruiser
            args:
                scores [list of floats]: runner and bruiser scores of ind. to choose parents
        """
        # Retrieve who is the runner and who is the bruiser
        runner_genes = np.argmax(scores[:, 0])
        bruiser_genes = np.argmax(scores[:, 1])
        R = int(scores[runner_genes, 2])
        B = int(scores[bruiser_genes, 3])
        # The miracle of life
        child = np.zeros([2, self.num_turns, 2])
        child[R, :, :] = self.pop[R, runner_genes, :, :]
        child[B, :, :] = self.pop[B, bruiser_genes, :, :]

        return child

    def mutate(self, base, amplitude, scores, exclude):
        """ Mutate individuals
            Only mutate one attribute (angle or thrust or shield) per individual 
            This ensures good mutations in one place aren't lost by a bad mutations elsewhere

            args:
                base [np.array of individuals]: individuals to mutate
                amplitude [int]: Magnitude of mutations (should decrease with generations)
                scores [list of floats]: runner and bruiser scores of ind. to better guide mutations
                exclude [int]: index of base population to not mutate
        """
        # Retrieve who is the runner and who is the bruiser
        n = self.pop_size
        runner_genes = np.argmax(scores[:, 0])
        bruiser_genes = np.argmax(scores[:, 1])
        R = int(scores[runner_genes, 2])
        B = int(scores[bruiser_genes, 3])

        # Choose to either update runner or bruiser (since runner mistakes are much more costly, update it more to decrease likelihood of not finding good move)
        i = np.random.choice([R, B], p=[0.75, 0.25])
        if i == R:
            move_prob = 0.85
            shield_prob = 0.6
        else:
            move_prob = 0.7
            shield_prob = 0.8
        # Choose to update angles/speed or shield
        if np.random.rand() < move_prob:
            # Move mutation
            turn_choices = [range(self.num_turns)] + [[0,1], [2,3], [0,1,2], [1,2,3]] 
            for j in xrange(n):
                if j == exclude:
                    continue
                turns_to_update = turn_choices[np.random.randint(0, len(turn_choices))]
                # Choose to update angles or speed
                if np.random.rand() < 0.75:
                    base[i, j, turns_to_update, 0] += np.random.choice([1,-1]) * amplitude * 0.628319
                else:
                    base[i, j, turns_to_update, 1] += np.random.choice([1,-1]) * amplitude * 100
                    base[i, j, turns_to_update, 1] = np.clip(base[i, j, turns_to_update, 1], 10, 100)

            base[i,:,:,0] = np.clip(base[i,:,:,0], -0.314159, 0.314159)
        else:
            # Shield mutation
            give_shield_turn = (np.random.rand(2, n) < shield_prob)
            for j in xrange(n):
                if j == exclude:
                    continue
                if give_shield_turn[i][j] and (abs(self.pods.vxy[R][0]) + abs(self.pods.vxy[R][0])) > 200:
                    shield_turn = np.random.randint(self.num_turns)
                    base[i, j, shield_turn, 1] = -1
        

    def evaluate(self, my_sim, pods):
        """ Fitness Function
            args:
                my_sim [Bool]: True if evolving my pods, False if evolving enemy pods
                pods [individual]: individual to evaluate (score depends on all pods)
            NOTE: fitness and score used interchangeably
        """ 
        # Determine runner (R) and bruiser (B) pods
        scores = pods.evaluate_pos()
        R = np.argmax(scores[:2]) if my_sim else np.argmax(scores[2:])
        enemy_R = np.argmax(scores[2:]) + 2 if my_sim else np.argmax(scores[:2])
        B = int(not R)
        # runner prime (Rp) and bruiser prime (Bp) take into account who is being simulated
        Rp, Bp = (R, B) if my_sim else (R + 2, B + 2)
        
        # Ratio distance to next checkpoint for runner pod
        d_cp = max(0., 3.e6 - sum((pods.xy[Rp] - pods.next_cp[Rp])**2)) / 3.e6
        # Runner fitness determined it's absolute score (retrieved from Pods) minus it's use of shields (since shields prevent acceleration)
        # minus the angle between itself and it's next checkpoint (scaled so that when it's close to it's next checkpoint, this angle doesn't matter much)
        # minus the angle between itself and it's next next checkpoint (scaled so that when it's close to it'se next checkpoint, this matters more)
        runner_score = scores[Rp] - 5e9 * pods.shields[Rp] - \
                       abs(diff_angle(pods.xy[Rp], pods.checkpoints[(pods.cp_id[Rp])], pods.angle[Rp])) * (1.e6) * (1. - d_cp) - \
                       abs(diff_angle(pods.xy[Rp], pods.checkpoints[(pods.cp_id[Rp] + 1) % pods.num_cp], pods.angle[Rp])) * d_cp * (5.e5 + (abs(pods.vxy[Rp][0]) + abs(pods.vxy[Rp][1]) * 500))
        # Bruiser fitness determined by the the negative of the enemy runner's absolute score minus the distance squared between itself and the enemy runner
        # minus the distance squared between itself and the enemy runner's next checkpoint
        # minus how hard the collisions between itself and the runner are
        # plus how hard the collisions with shield on happen between itself and the enemy
        bruiser_score = (-1) * (scores[enemy_R] + sum((pods.xy[Bp] - pods.next_cp[enemy_R])**2) + sum((pods.xy[Bp] - pods.xy[enemy_R])**2))
        bruiser_score -= pods.self_hit * 3e4 
        bruiser_score += pods.shield_hit[Bp] * 2.e4

        # Reset hit counters
        pods.self_hit = 0
        pods.shield_hit = 0.

        return runner_score, bruiser_score, R, B

    def simulate(self, my_sim, moves):
        """ Simulate individual to get it's fitness
            args:
                my_sim [Bool]: True if evolving my pods, False if evolving enemy pods
                moves [np array]: individual to simulate
        """
        pods = copy.deepcopy(self.pods)
        for t in xrange(self.num_turns):
            pods.play(moves[:, t, :])

        return self.evaluate(my_sim, pods)

    def move_order(self, my_sim, evolu_pod_moves, other_pod_moves):
        """ Helper function to map the current population (2) and estimated others (2) to the simulation moves (4) """
        if my_sim:
            return np.concatenate((evolu_pod_moves, other_pod_moves))
        else:
            return np.concatenate((other_pod_moves, evolu_pod_moves))

    def evolve(self, my_sim, max_time, prev_best, other_pod_moves):
        """ 
            Evolution of pod movements
            An individual (ind.) is a sequence of moves represented by a np array with shape = (number of pods we control, population size, number of turns, 2)
            where number of pods we control = 2
            and the final dimension are the controls for each pod at every turn.
            The controls of each pod are:
                1. a change from current angle bounded by (-18 deg, 18 deg) 
                2. a thrust bounded by (0, 100)
                Note: The thrust can be replaced by -1 to activate a shield which increases mass by 10, but prevents any thrust for the next 3 turns
    
            Each generation:
                copy best pod
                breed a new pod by combining the best runner from one individual with the best bruiser of another
                mutate pods except for newly bred pod
                simulate pods to evaluate them
                replace worst pod with previously copied best pod
    
            args:
                my_sim [Bool]: True if evolving my pods, False if evolving enemy pods
                max_time [int]: Amount of time to run evolution for
                prev_best [individual]: Best individual from last evolution
                other_pod_moves [individual]: Estimated moves of other 2 pods
        """
        s_time = time.time() * 1000
        # Generate population
        self.pop = self.generate_population(my_sim)
        # Use best ind. of previous iteration as one of the starting individuals
        self.pop[:, 0] = prev_best
        # Get fitness of population
        split_scores = np.array([self.simulate(my_sim, self.move_order(my_sim, self.pop[:,i,:,:], other_pod_moves)) for i in range(self.pop_size)])
        scores = np.array(map(lambda x: x[0] + x[1], split_scores))
        
        gen = 1
        # For alloted time
        while time.time() * 1000 - s_time < max_time:
            # Replace worse performing ind. with child
            worst_pod = np.argmin(scores)
            self.pop[:, worst_pod] = self.breed(split_scores)
            

            # Elitism - keep best current ind. (use runner score for choosing since runner mistakes are much more costly) for next gen
            best_pod = np.argmax(split_scores[:, 0])
            elite, elite_score = copy.deepcopy(self.pop[:, best_pod]), scores[best_pod]

            # Mutate population (except new child)
            self.mutate(self.pop, min(max(0.1, (17-gen)/15.), 1.0), split_scores, worst_pod)

            # Get fitness of new population
            split_scores = np.array([self.simulate(my_sim, self.move_order(my_sim, self.pop[:,i,:,:], other_pod_moves)) for i in range(self.pop_size)])
            scores = np.array(map(lambda x: x[0] + x[1], split_scores))

            # Replace worst ind. with best ind. of previous generation
            worst_pod = np.argmin(scores)
            self.pop[:, worst_pod] = elite
            scores[worst_pod] = elite_score

            gen += 1

        # Return best performing pod
        return self.pop[:, np.argmax(scores)], gen, np.max(scores)# + min_score - 1

        
    def run(self):
        """ Run genetic algorithm """
        # Evolve enemy pods first for a bit to estimate their trajectory
        enemy_best, _a, _x = self.evolve(False, 20, self.enemy_prev_best, self.my_prev_best)
        # Save the best sequence of moves to use as one of the starting population in the next iteration
        self.enemy_prev_best = np.concatenate((enemy_best[:, 1:], enemy_best[:, -1:]), axis=1)

        # Evolve my own pods to find best path
        my_best, _b, _y = self.evolve(True, 95, self.my_prev_best, enemy_best)
        # Save the best sequence of moves to use as one of the starting population in the next iteration
        self.my_prev_best = np.concatenate((my_best[:, 1:], my_best[:, -1:]), axis=1)

        # Translate the best first move from my simulation to game input
        angle = my_best[:, 0, 0]
        thrust = my_best[:, 0, 1]
        target_x, target_y = self.pods.get_target(np.concatenate((angle, np.zeros(2))))
        for i in xrange(2):
            thrust_val = 'SHIELD' if thrust[i] == -1 else int(thrust[i])
            print int(target_x[i]), int(target_y[i]), thrust_val

        # Return only used for testing
        return my_best, enemy_best, _a, _b

# Get input from game
checkpoints = []
num_laps = int(raw_input())
for num_cp in xrange(int(raw_input())):
    checkpoints.append([int(i) for i in raw_input().split()])

base_pods = Pods(np.array(checkpoints), num_laps)
evol = Evolution(base_pods)
first_round = True
# Game iterations
while True:
    # Each pod gets an updated x, y, vx, vy, angle, and next_checkpoint id
    # First two pods are mine
    # Second 2 pods are the enemies
    p1 = [int(i) for i in raw_input().split()]
    p2 = [int(i) for i in raw_input().split()]
    p3 = [int(i) for i in raw_input().split()]
    p4 = [int(i) for i in raw_input().split()]
    base_pods.update_pod(*[[p1[i]] + [p2[i]] + [p3[i]] + [p4[i]] for i in range(6)])       
    
    # For some reason the angle given on the first round is -1
    # However we always start facing the first checkpoint, so boost it straight on first round
    if first_round:
        dxy = np.array(checkpoints[0]) - np.array(checkpoints[1])

        angle = np.arctan2(dxy[0], -dxy[1]) + PI/2.

        target_x, target_y = base_pods.xy[:, 0] + np.cos(angle) * 10000., base_pods.xy[:, 1] + np.sin(angle) * 10000.
        for i in xrange(2):
            print int(target_x[i]), int(target_y[i]), 'BOOST'
        first_round=False
    else:
        # If not first round, run genetic algorithm
        evol.run()



######################## BELOW CODE IS FOR PERSONAL TESTS ###################################
# def test():
#     final_scores = []
#     final_gens = []

#     best_path = None
#     overall_best_score = 0

#     # for i in xrange(5, 20):
#     #     best_score = 0
#     #     for j in xrange(250):
#     checkpoints = [(0,0), (2500,2500)]
#     num_laps = 2
#     base_pods = Pod(checkpoints, num_laps)
    

#     base_pods.update_pod(np.zeros(4),np.zeros(4),np.zeros(4),np.zeros(4),np.zeros(4),np.ones(4, dtype=int))

#     x = [[], [], [], []]
#     y = [[], [], [], []]
#     scores = []

#     for _ in xrange(40):
#         evol = Evolution(base_pods)
        
#         champion, sith_lord, my_gens, enemy_gens = evol.run()

#         final_score, new_x, new_y = evol.simulate(True, np.concatenate([champion, sith_lord]), extra=True) # 
#         scores.append(final_score)
#         # print new_x
#         for i in xrange(4):
#             x[i].append(new_x[i][0])
#             y[i].append(new_y[i][0])
#         # print x, y
#         evol.pods.play(np.concatenate((champion[:, 0], sith_lord[:, 0])))

        
#     plt.figure(1)
#     plt.plot(x[0], y[0], 'b')
#     plt.plot(x[1], y[1], 'g')
#     plt.plot(x[2], y[2], 'r-')
#     plt.plot(x[3], y[3], 'ro')
#     plt.figure(2)
#     plt.plot(range(len(scores)), scores)
#     # print final_score, my_gens, enemy_gens

#     # if final_score > best_score:
#     #     best_score = final_score

#     # if final_score > overall_best_score:
#     #     best_path = x, y
#     #     overall_best_score = final_score

#     # final_scores.append(final_score)
#     # final_gens.append(num_gens)

#     # output = i, best_score, np.mean(final_scores), np.var(final_scores), np.mean(final_gens), np.var(final_gens)
#     # print ','.join([str(o) for o in output])
#     # plt.plot(range(len(scores)), scores)
#     plt.show()



# # # for i in xrange(10):
# test()
