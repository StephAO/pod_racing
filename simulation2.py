import numpy as np
# import matplotlib.pyplot as plt
import time
import copy
import sys

PI = np.pi
TWO_PI = 2* np.pi

class Pod:
    def __init__(self, checkpoints, req_laps):
        self.x, self.y = 0, 0
        self.vx, self.vy = 0., 0.
        self.angle = 0
        self.shield = False
        self.boost_used = True
        self.checkpoints = checkpoints
        self.cp_id = 0
        self.num_cp = len(checkpoints)
        self.previous_best = None
        self.lap = 0
        self.next_cp = None
        self.cp_passed = 0
        self.req_laps = req_laps

    def update_pod(self, x, y, vx, vy, angle, cp_id):
        self.x, self.y = x, y
        self.vx, self.vy = vx, vy
        self.angle = angle*PI/180.
        if cp_id != self.cp_id % self.num_cp:
            self.cp_passed += 1
            self.cp_id = cp_id % self.num_cp
            if cp_id == 1:
                self.lap += 1 
        self.next_cp = self.checkpoints[self.cp_id]

    def dist(self, othr):
        return np.sqrt((self.x - othr.x)**2 + (self.y - othr.y)**2)

    def dist2(self, othr):
        return (self.x - othr.x)**2 + (self.y - othr.y)**2

    def dist2(self, x2, y2):
        return (self.x - x2)**2 + (self.y - y2)**2

    def get_angle_diff(self, x2, y2):
        if x2 == self.x and y2 == self.y:
            return 0.
        dx = x2 - self.x
        dy = y2 - self.y

        return (np.arctan2(dy, dx)) + PI

    def turn_angle(self, othr):
        target_angle = self.get_angle_diff(othr)

        cw = self.angle + TWO_PI - target_angle if target_angle > self.angle else self.angle - target_angle
        
        ccw = TWO_PI - cw
        rotation = -min(cw, 0.314159) if cw < ccw else min(ccw, 0.314159)
        return rotation

    def rotate(self, angle):
        self.angle += angle
        
        if self.angle >= TWO_PI:
            self.angle -= TWO_PI
        if self.angle < 0.:
            self.angle += TWO_PI

    def accelerate(self, thrust):
        if self.shield:
            return
        self.vx -= np.cos(self.angle) * thrust
        self.vy -= np.sin(self.angle) * thrust

    def move(self, t):
        self.x += self.vx * t
        self.y += self.vy * t

    def end(self, simulation=True): 
        self.x = round(self.x)
        self.y = round(self.y)
        self.vx = int(self.vx * 0.85)
        self.vy = int(self.vy * 0.85)
        if self.dist2(*self.next_cp) < 600**2:
            self.cp_id = (self.cp_id + 1) % self.num_cp
            if self.cp_id == 1:
                self.lap += 1 
            self.cp_passed += 1
            self.next_cp = self.checkpoints[self.cp_id]
        return False

    def play(self, move, simulation=True):
        self.rotate(move[0])
        self.accelerate(move[1])
        self.move(1.0)
        self.end(simulation=simulation)

    def evaluate_pos(self):
        return self.cp_passed*1e9 + max(0, (337e6 - self.dist2(*self.next_cp))) + 1e20 * (self.lap == self.req_laps)

    def get_target(self, angle):
        target_angle = self.angle + angle

        if target_angle >= TWO_PI:
            target_angle -= TWO_PI
        if target_angle < 0.:
            target_angle += TWO_PI
        return self.x - np.cos(target_angle) * 10000., self.y - np.sin(target_angle) * 10000.

    def __str__(self):
        return "(%f, %f) angled at %f with velocities (%f, %f)" % (self.x, self.y, self.angle * 180 / PI, self.vx, self.vy)

    def __eq__(self, othr):
        return (self.x == othr.x and self.y == othr.y)

class Evolution:
    def __init__(self, pods, pop_size=7, num_turns=5):
        self.pods = pods
        self.start_state = None
        self.pop = None
        self.pop_size = pop_size
        self.num_turns = num_turns
        self.my_prev_best = np.zeros([2, self.num_turns, 2])
        self.enemy_prev_best = np.zeros([2, self.num_turns, 2])

    def generate_population(self):
        # Create population with random initialization
        # Each individual consists of n turns with each turn having an angle and thrust associated with it.
        pop = np.zeros([2, self.pop_size, self.num_turns, 2])
        pop[:,1,:,0] = np.random.rand(2, self.num_turns) * 0.314159
        pop[:,2,:,0] = np.random.rand(2, self.num_turns) * -0.314159
        pop[:,3,:,0] = np.random.rand(2, self.num_turns) * 0.1 - 0.05
        pop[:,4:,:,0] = np.random.rand(2, self.pop_size-4, self.num_turns) * 0.628319 - 0.314159
        pop[:,:,:,1] = np.random.randint(50, 101, (2, self.pop_size, self.num_turns))
        return pop

    def breed(self, scores, num_children, num_parents):
        # Select parents based on the distribution of their score
        parents = self.pop[:, np.random.choice(range(self.pop_size), size=(num_parents), replace=False, p=(scores/np.sum(scores)))]

        # Create children by randomly selecting attributes from parents
        children = np.zeros([2, num_children, self.num_turns, 2])
        children[0,:,:,0] = parents[0, np.random.randint(num_parents, size=(num_children, self.num_turns)), range(self.num_turns), 0]
        children[0,:,:,1] = parents[0, np.random.randint(num_parents, size=(num_children, self.num_turns)), range(self.num_turns), 1]
        children[1,:,:,0] = parents[1, np.random.randint(num_parents, size=(num_children, self.num_turns)), range(self.num_turns), 0]
        children[1,:,:,1] = parents[1, np.random.randint(num_parents, size=(num_children, self.num_turns)), range(self.num_turns), 1]

        return children

    def mutate(self, base, amplitude):
        # Randomly increase or decrease amplitude
        base[:,:,:,0] += np.random.choice([1,-1]) * np.random.rand(2, self.pop_size, self.num_turns) * amplitude * 0.628319
        base[:,:,:,1] += np.random.choice([1,-1]) * np.random.rand(2, self.pop_size, self.num_turns) * amplitude * 100.
        # print >> sys.stderr, amplitude, base[:,:,:,1]
        base[:,:,:,0] = np.clip(base[:,:,:,0], -0.314159, 0.314159)
        base[:,:,:,1] = np.clip(base[:,:,:,1], 0, 100)

    def evaluate(self, pods):
        my_scores = [pod.evaluate_pos() for pod in pods[:2]]
        enemy_scores = [pod.evaluate_pos() for pod in pods[2:]]
        my_runner = np.argmax(my_scores)
        enemy_runner = np.argmax(enemy_scores)


        return sum(my_scores)

    def simulate(self, my_sim, moves):
        pods = copy.deepcopy(self.pods)
        for t in xrange(self.num_turns):
            if my_sim:
                pods[0].play(moves[0][t])
                pods[1].play(moves[1][t])
                pods[2].play(moves[2][t])
                pods[3].play(moves[3][t])
            else:
                pods[0].play(moves[2][t])
                pods[1].play(moves[3][t])
                pods[2].play(moves[0][t])
                pods[3].play(moves[1][t])
        score = self.evaluate(pods)
        

        return score

    def evolve(self, my_sim, max_time, prev_best, other_pod_moves, num_children, num_parents):
        self.pop = self.generate_population()
        self.pop[:, 0] = prev_best
        
        scores = np.array([self.simulate(my_sim, np.concatenate((self.pop[:,i,:,:], other_pod_moves))) for i in range(self.pop_size)])
        min_score = np.min(scores)
        scores -= min_score - 1
        s_time = time.time() * 1000
        gen = 1
        # gen_scores = []
        
        while time.time() * 1000 - s_time < max_time:
            prev_min_score = min_score

            # Replace worse performing pod_movess with children
            worst_two_ind = np.argpartition(scores, num_children)[:num_children]
            self.pop[:, worst_two_ind] = self.breed(scores, num_children=num_children, num_parents=num_parents)
            

            # Elitism - keep best current gen for next gen
            best_pod = np.argmax(scores)
            elite, elite_score = copy.deepcopy(self.pop[:, best_pod]), scores[best_pod] + prev_min_score - 1

            # Mutate pod_moves
            self.mutate(self.pop, amplitude = min(max(0.1, (50.-gen)/50.), 1.0))

            scores = np.array([self.simulate(my_sim, np.concatenate((self.pop[:,i,:,:], other_pod_moves))) for i in range(self.pop_size)])
            worst_pod = np.argmin(scores)
            self.pop[:, worst_pod] = elite
            scores[worst_pod] = elite_score

            min_score = np.min(scores)
            scores -= min_score - 1

            gen += 1

        # Return best performing pod
        return self.pop[:, np.argmax(scores)], gen, np.max(scores)

        
    def run(self, num_children=2, num_parents=3):
        # Simulate enemy pods first
        enemy_best, _a, _x = self.evolve(False, 20, self.enemy_prev_best, self.my_prev_best, num_children, num_parents)
        self.enemy_prev_best = np.concatenate((enemy_best[:, 1:], np.zeros((2, 1, 2))), axis=1)

        my_best, _b, _y = self.evolve(True, 110, self.my_prev_best, enemy_best, num_children, num_parents)
        self.my_prev_best = np.concatenate((my_best[:, 1:], np.zeros((2, 1, 2))), axis=1)

        for i in xrange(2):
            angle, thrust = my_best[i, 0]
            print >> sys.stderr, _a, _b, angle*180/PI, thrust, self.pods[i].x, self.pods[i].y, self.pods[i].cp_id, self.pods[i].next_cp
            target_x, target_y = self.pods[i].get_target(angle)
            print int(target_x), int(target_y), int(thrust)

        print >> sys.stderr, my_best, self.evaluate(self.pods), _y


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

first_loop = True
base_pods = [Pod(checkpoints, num_laps), Pod(checkpoints, num_laps), Pod(checkpoints, num_laps), Pod(checkpoints, num_laps)]
evol = Evolution(base_pods)
while True:
    # next_checkpoint_x: x position of the next check point
    # next_checkpoint_y: y position of the next check point
    # next_checkpoint_dist: distance to the next checkpoint
    # next_checkpoint_angle: angle between your pod orientation and the direction of the next checkpoint
    for pod in base_pods:
        pod.update_pod(*[int(i) for i in raw_input().split()])

    # print >> sys.stderr, str(pod.x), str(pod.y), str(pod.angle * 180/PI)
    # print >> sys.stderr, "***", pod.get_angle_diff(next_checkpoint_x, -next_checkpoint_y)*180/PI, next_checkpoint_angle        
    
    evol.run()

