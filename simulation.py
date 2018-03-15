import numpy as np
# import matplotlib.pyplot as plt
from scipy.spatial import distance
import time
import copy
import sys


def closest(a, b, me):
    da = b[1] - a[1]
    db = a[0] - b[0]
    c1 = da*a[0] + db*a[1]
    c2 = -db*me[0] + da*me[1]
    det = da*da + db*db

    if det:
        cx = (da*c1 - db*c2) / det
        cy = (da*c2 + db*c1) / det
    else:
        cx, cy = me

    return np.array([cx, cy])

def collision(xy1, xy2, vxy1, vxy2):
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


PI = np.pi
TWO_PI = 2* np.pi

class Pod:
    def __init__(self, checkpoints, req_laps):
        self.xy = np.zeros((4, 2)), np.zeros((4, 2))
        self.vxy = np.zeros((4, 2)), np.zeros((4, 2))
        self.angle = np.zeros((4))
        self.shield = np.array([False] * 4)
        self.self_hit = 0
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
        self.xy = np.concatenate((np.expand_dims(np.array(x, dtype=float), axis=1), np.expand_dims(np.array(y, dtype=float), axis=1)), axis=1)
        self.vxy = np.concatenate((np.expand_dims(np.array(vx, dtype=float), axis=1), np.expand_dims(np.array(vy, dtype=float), axis=1)), axis=1)
        self.angle = np.array(angle, dtype=float)*PI/180.
        cp_id = np.array(cp_id)
        if any(cp_id != self.cp_id % self.num_cp):
            self.cp_passed[cp_id != self.cp_id] += 1
            self.cp_id = cp_id % self.num_cp
            self.lap[cp_id == 1] += 1 
        self.next_cp = self.checkpoints[self.cp_id]

    def dist_to_cp(self):
        a = (self.xy - self.next_cp)**2 # + (self.y - self.next_cp[:, 1])**2
        return a[:, 0] + a[:, 1]


    def getAngle(self, p):
        d = np.sqrt(np.sum((self.xy - p)**2, axis=1))
        dxy = (p - self.xy) / np.expand_dims(d, axis=1)
        a = np.arccos(dxy[:, 0])
        a[(dxy[:, 1] < 0)] = TWO_PI - a[(dxy[:, 1] < 0)]

        return a;

    def diffAngle(self, p):
        a = self.getAngle(p);

        right = a - self.angle
        idx = (self.angle > a)
        right[idx]  = TWO_PI - self.angle[idx] + a[idx]
        left = self.angle - a
        idx = (self.angle < a)
        left[idx] = self.angle[idx] + 360.0 - a[idx]

        da = -left
        da[right < left] = right[right < left]
        return da

    def bounce(self, i, j):
        dxy = self.xy[i, :] - self.xy[j, :]
        dv = self.vxy[i, :] - self.vxy[j, :]
        m1, m2 = self.shield[i] * 9. + 1., self.shield[j] * 9. + 1.
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
        self.angle = (self.angle + angle) % TWO_PI

    def accelerate(self, thrust):
        not_shield = np.logical_not(self.shield)
        self.vxy[not_shield, 0] += np.cos(self.angle[not_shield]) * thrust
        self.vxy[not_shield, 1] += np.sin(self.angle[not_shield]) * thrust

    def move(self):
        t = 0.
        # while t < 1.0:
        collisions = [(i, j, collision(self.xy[i], self.xy[j], self.vxy[i], self.vxy[j])) for i in xrange(4) for j in xrange(4) if i != j]
        c = min(collisions, key=lambda x: x[2] or 2.)

        if c[2] is None:
            t = 1
            self.xy += self.vxy * 1.0
        else:
            t = c[2]
            self.xy += self.vxy * c[2]
            if c[0] == c[1]:
                self.self_hit += 1
            self.bounce(c[0], c[1])
            self.xy += self.vxy * (1. - c[2])


    def end(self, simulation=True): 
        self.xy = np.round(self.xy)
        self.vxy = (self.vxy * 0.85).astype(int).astype(float)

        # Update checkpoints if checkpoints are reached
        # reduced 360000 to 300000 to make sure i hit the cp
        reached_cp = self.dist_to_cp() < 160000
        if any(reached_cp):
            self.cp_id[reached_cp] = (self.cp_id[reached_cp] + 1) % self.num_cp
            for i in xrange(4):
                if reached_cp[i] and self.cp_id[i] == 1:
                    self.lap[i] += 1 
            self.cp_passed[reached_cp] += 1
            self.next_cp[reached_cp] = self.checkpoints[self.cp_id[reached_cp]]

    def play(self, move, simulation=True):
        self.rotate(move[:, 0])
        self.accelerate(move[:, 1])
        self.move()
        self.end(simulation=simulation)

    def evaluate_pos(self):
        return self.cp_passed*1e10 - self.dist_to_cp() #+ 1e20 * (self.lap == self.req_laps)

    def get_target(self, angle):
        target_angle = (self.angle + angle) % TWO_PI

        return self.xy[:, 0] + np.cos(target_angle) * 10000., self.xy[:, 1] + np.sin(target_angle) * 10000.

    def __str__(self):
        return "(%f, %f) angled at %f with velocities (%f, %f)" % (self.x, self.y, self.angle * 180 / PI, self.vx, self.vy)

    def __eq__(self, othr):
        return (self.x == othr.x and self.y == othr.y)







class Evolution:
    def __init__(self, pods, pop_size=4, num_turns=4):
        self.pods = pods
        self.start_state = None
        self.pop = None
        self.pop_size = pop_size
        self.num_turns = num_turns
        self.my_prev_best = np.zeros([2, self.num_turns, 2])
        self.enemy_prev_best = np.zeros([2, self.num_turns, 2])

        self.my_prev_best[:,:,1] = 100
        self.enemy_prev_best[:,:,1] = 100

    def generate_population(self):
        # Create population with random initialization
        # Each individual consists of n turns with each turn having an angle and thrust associated with it.
        pop = np.zeros([2, self.pop_size, self.num_turns, 2])
        # CHILD 1 and 2 BRUISER Angles
        scores = self.pods.evaluate_pos()
        R = np.argmax(scores[:2])
        print >> sys.stderr, scores, " - ", R
        B = int(not R)
        B_diff_angles = self.pods.diffAngle(self.pods.next_cp[np.argmax(scores[2:] + 2)])
        B_max_num_turns = (np.abs(B_diff_angles) / 0.314159).astype(int)
        # CHILD 1 RUNNER ANGLES
        R_diff_angles = self.pods.diffAngle(self.pods.next_cp[R])
        R_max_num_turns = (np.abs(R_diff_angles) / 0.314159).astype(int)
        R_turning_angle = R_diff_angles / (R_max_num_turns + 1)
        pop[R, 1, :R_max_num_turns[R] + 1, 0] = R_turning_angle[R]
        pop[B, 1, :B_max_num_turns[B] + 1, 0] = (B_diff_angles / (B_max_num_turns + 1))[B]
        # CHILD 2 RUNNER ANGLES
        R_diff_angles = self.pods.diffAngle(self.pods.checkpoints[(self.pods.cp_id + 1) % self.pods.num_cp])
        R_max_num_turns = (np.abs(R_diff_angles) / 0.314159).astype(int)
        R_turning_angle = R_diff_angles / (R_max_num_turns + 1)
        pop[R, 2, :R_max_num_turns[R] + 1, 0] = R_turning_angle[0]
        pop[B, 2, :B_max_num_turns[B] + 1, 0] = (B_diff_angles / (B_max_num_turns + 1))[B]
        # pop[:,2,:,0] = 0.
        pop[:,3:,:,0] = np.random.rand(2, self.pop_size-3, self.num_turns) * 0.628319 - 0.314159
        # pop[:,4:,:,0] = np.random.rand(2, self.pop_size-4, self.num_turns) * 0.628319 - 0.314159
        pop[:,:,:,1] = np.random.randint(50, 101, (2, self.pop_size, self.num_turns))
        return pop

    def breed(self, scores):
        # Select parents based on the distribution of their score
        runner_genes = np.argmax(scores[:, 0])
        bruiser_genes = np.argmax(scores[:, 1])
        R = int(scores[runner_genes, 2])
        B = int(scores[bruiser_genes, 3])

        # print >> sys.stderr, R, B, runner_genes, bruiser_genes, scores.shape
        child = np.zeros([2, 1, self.num_turns, 2])
        child[R, 0, :, :] = self.pop[R, runner_genes, :, :]
        child[B, 0, :, :] = self.pop[B, bruiser_genes, :, :]

        return child

    def mutate(self, base, amplitude):
        # Randomly increase or decrease amplitude
        turn_choices = range(self.num_turns) + [range(self.num_turns)] + [[0,1], [2,3]]
        for i in xrange(self.pop_size):
            turns_to_update = turn_choices[np.random.randint(0, len(turn_choices))]
            P = np.random.choice([0, 1])
            base[P, i, turns_to_update, 0] = np.random.choice([1,-1]) * amplitude * 0.628319
            base[P, i, turns_to_update, 1] = np.random.choice([1,-1]) * amplitude * 100
        
        base[:,:,:,0] = np.clip(base[:,:,:,0], -0.314159, 0.314159)
        base[:,:,:,1] = np.clip(base[:,:,:,1], 0, 100)

    def evaluate(self, my_sim, pods):
        scores = pods.evaluate_pos()
        my_runner = np.argmax(scores[:2])
        enemy_runner = np.argmax(scores[2:]) + 2
        my_bruiser = int(not my_runner)
        # enemy_bruiser = int(not enemy_runner) + 2
        # enemy_runner += 2
        runner_score = scores[my_runner]
        bruiser_score = (-1) * (scores[enemy_runner] + sum((pods.xy[my_bruiser] - pods.next_cp[enemy_runner])**2) + sum((pods.xy[my_bruiser] - pods.xy[enemy_runner])**2))
        # final_score -= pods.self_hit * 3e8
        # pods.self_hit = 0
        # final_score +=  pods.sd[my_runner][enemy_bruiser]
        # my_runner = np.argmax(my_scores)
        # enemy_runner = np.argmax(enemy_scores)

        return runner_score, bruiser_score, my_runner, my_bruiser

    def simulate(self, my_sim, moves, extra=False):
        pods = copy.deepcopy(self.pods)
        # for test
        # x = [[],[],[],[]]
        # y = [[],[],[],[]]
        #
        for t in xrange(self.num_turns):
            pods.play(moves[:, t, :])

            # for test
            # for i in xrange(4):
            #     x[i].append(pods.xy[i, 0])
            #     y[i].append(pods.xy[i, 1])
            # #
        # score = self.evaluate(my_sim, pods)
        
        # if extra:
        #     return score, x, y
        # else:
        #     
        return self.evaluate(my_sim, pods)

    def move_order(self, my_sim, evolu_pod_moves, other_pod_moves):
        if my_sim:
            return np.concatenate((evolu_pod_moves, other_pod_moves))
        else:
            return np.concatenate((other_pod_moves, evolu_pod_moves))

    def evolve(self, my_sim, max_time, prev_best, other_pod_moves, num_children, num_parents):
        self.pop = self.generate_population()
        self.pop[:, 0] = prev_best

        split_scores = np.array([self.simulate(my_sim, self.move_order(my_sim, self.pop[:,i,:,:], other_pod_moves)) for i in range(self.pop_size)])
        scores = np.array(map(lambda x: x[0] + x[1], split_scores))

        min_score = np.min(scores)
        scores -= min_score - 1
        s_time = time.time() * 1000
        gen = 1
        # gen_scores = []
        
        while time.time() * 1000 - s_time < max_time:
            prev_min_score = min_score

            # Replace worse performing pod_movess with children
            worst_two_ind = np.argpartition(scores, num_children)[:num_children]
            self.pop[:, worst_two_ind] = self.breed(split_scores)
            

            # Elitism - keep best current gen for next gen
            best_pod = np.argmax(scores)
            elite, elite_score = copy.deepcopy(self.pop[:, best_pod]), scores[best_pod] + prev_min_score - 1

            # Mutate pod_moves
            self.mutate(self.pop, amplitude = min(max(0.1, (20-gen)/10.), 1.0))

            split_scores = np.array([self.simulate(my_sim, self.move_order(my_sim, self.pop[:,i,:,:], other_pod_moves)) for i in range(self.pop_size)])
            scores = np.array(map(lambda x: x[0] + x[1], split_scores))

            worst_pod = np.argmin(scores)
            self.pop[:, worst_pod] = elite
            scores[worst_pod] = elite_score

            min_score = np.min(scores)
            scores -= min_score - 1

            gen += 1

        # Return best performing pod
        return self.pop[:, np.argmax(scores)], gen, np.max(scores) + min_score - 1

        
    def run(self, num_children=1, num_parents=2):
        # Simulate enemy pods first
        enemy_best, _a, _x = self.evolve(False, 20, self.enemy_prev_best, self.my_prev_best, num_children, num_parents)
        self.enemy_prev_best = np.concatenate((enemy_best[:, 1:], enemy_best[:, -1:]), axis=1)

        my_best, _b, _y = self.evolve(True, 100, self.my_prev_best, enemy_best, num_children, num_parents)
        self.my_prev_best = np.concatenate((my_best[:, 1:], my_best[:, -1:]), axis=1)

        angle = my_best[:, 0, 0]
        thrust = my_best[:, 0, 1]
        # print >> sys.stderr, _a, _b, angle*180/PI, thrust, self.pods[i].x, self.pods[i].y, self.pods[i].cp_id, self.pods[i].next_cp
        target_x, target_y = self.pods.get_target(np.concatenate((angle, np.zeros(2))))
        
        for i in xrange(2):
            print int(target_x[i]), int(target_y[i]), int(thrust[i])

        print >> sys.stderr, _a, _b

        return my_best, enemy_best, _a, _b


    # def print_next_output(self, next_checkpoint_dist, next_checkpoint_angle):
    #     if self.previous_best is not None and all(np.abs(self.previous_best[:2, 0]) > 0.3):
    #         best_moves, num_gen = self.previous_best, 0
    #     else:
    #         evol = Evolution(self, pop_size=10)
    #         best_moves, num_gen = evol.run(previous_best=self.previous_best, num_children=4, num_parents=4)
            
    #     angle, thrust = best_moves[0]
    #     self.previous_best = np.concatenate((best_moves[1:], np.array([[0.,50.]])))

    #     # print >> sys.stderr, str(angle * 180/PI), str(thrust), str(num_gen)


    #     if not self.boost_used and next_checkpoint_dist > 5000 and abs(next_checkpoint_angle) < 10:
    #         thrust = "BOOST"
    #         self.boost_used = True
    #     else:
    #         thrust = int(thrust)

    #     for i in xrange(2):
    #         target_x, target_y = pod[i].get_target(angle)
    #         print int(target_x), int(-target_y), thrust[i]

checkpoints = []
num_laps = int(raw_input())
for num_cp in xrange(int(raw_input())):
    checkpoints.append([int(i) for i in raw_input().split()])

base_pods = Pod(np.array(checkpoints), num_laps)
evol = Evolution(base_pods)
first_round = True
while True:
    # next_checkpoint_x: x position of the next check point
    # next_checkpoint_y: y position of the next check point
    # next_checkpoint_dist: distance to the next checkpoint
    # next_checkpoint_angle: angle between your pod orientation and the direction of the next checkpoint
    p1 = [int(i) for i in raw_input().split()]
    p2 = [int(i) for i in raw_input().split()]
    p3 = [int(i) for i in raw_input().split()]
    p4 = [int(i) for i in raw_input().split()]
    base_pods.update_pod(*[[p1[i]] + [p2[i]] + [p3[i]] + [p4[i]] for i in range(6)])

    # print >> sys.stderr, str(pod.x), str(pod.y), str(pod.angle * 180/PI)
    # print >> sys.stderr, "***", pod.get_angle_diff(next_checkpoint_x, -next_checkpoint_y)*180/PI, next_checkpoint_angle        
    
    if first_round:
        dxy = np.array(checkpoints[0]) - np.array(checkpoints[1])

        angle = np.arctan2(dxy[0], -dxy[1]) + PI/2.

        target_x, target_y = base_pods.xy[:, 0] + np.cos(angle) * 10000., base_pods.xy[:, 1] + np.sin(angle) * 10000.
        for i in xrange(2):
            print int(target_x[i]), int(target_y[i]), 'BOOST'
        first_round=False
    else:
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
