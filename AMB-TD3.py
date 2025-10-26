import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
import random
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde


# Terrain Generator
class TerrainGenerator:
    def __init__(self, size, height_range, num_mountains):
        self.size = size
        self.height_range = height_range
        self.num_mountains = num_mountains
        self.terrain = self.generate_mountainous_terrain()

    # Generate Gaussian mounds
    def _gaussian_mound(self, x0, y0, sigma, amplitude):
        x = np.arange(self.size)
        y = np.arange(self.size)
        X, Y = np.meshgrid(x, y)
        return amplitude * np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma ** 2))

    # Superimpose Gaussian mounds and random noise to generate mountainous terrain
    def generate_mountainous_terrain(self):
        base = np.random.normal(0, 5, (self.size, self.size))
        terrain = np.zeros((self.size, self.size))
        for _ in range(self.num_mountains):
            x0 = np.random.randint(0.2 * self.size, 0.8 * self.size)
            y0 = np.random.randint(0.2 * self.size, 0.8 * self.size)
            sigma = np.random.uniform(80, 120)
            amplitude = np.random.uniform(100, 300)
            gaussian_mound = self._gaussian_mound(x0, y0, sigma, amplitude)
            terrain = np.maximum(terrain, gaussian_mound)
        terrain += base
        terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min())
        terrain = terrain * (self.height_range[1] - self.height_range[0]) + self.height_range[0]
        return terrain

    def get_elevation(self, x, y):
        x = np.clip(x, 0, self.size - 1)
        y = np.clip(y, 0, self.size - 1)
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        x1, y1 = min(x0 + 1, self.size - 1), min(y0 + 1, self.size - 1)
        z00 = self.terrain[y0, x0]
        z01 = self.terrain[y0, x1]
        z10 = self.terrain[y1, x0]
        z11 = self.terrain[y1, x1]
        dx, dy = x - x0, y - y0
        return (1 - dx) * (1 - dy) * z00 + dx * (1 - dy) * z01 + (1 - dx) * dy * z10 + dx * dy * z11

# Differential Evolution Algorithm (DE)
class DEAlgorithm:
    def __init__(self, pop_size, bounds, F=0.5, CR=0.7):
        self.pop_size = pop_size
        self.bounds = bounds
        self.F = F
        self.CR = CR
        self.dim = len(bounds)
        self.population = self.initialize_population()
        self.iteration = 0

    def initialize_population(self):
        population = []
        for _ in range(self.pop_size):
            individual = []
            for bound in self.bounds:
                individual.append(np.random.uniform(bound[0], bound[1]))
            population.append(np.array(individual))
        return np.array(population)

    def optimize(self, func, max_iter=10, verbose=True):
        best_fitness = float('inf')
        best_solution = None
        fitness_history = []

        for self.iteration in range(max_iter):
            fitness = np.array([func(ind) for ind in self.population])

            min_idx = np.argmin(fitness)
            current_best_fitness = fitness[min_idx]
            improvement = best_fitness - current_best_fitness if best_fitness != float('inf') else current_best_fitness

            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_solution = self.population[min_idx].copy()


            fitness_history.append(best_fitness)

            new_population = []
            for i in range(self.pop_size):
                candidates = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = self.population[np.random.choice(candidates, 3, replace=False)]

                # Mutation
                mutant = a + self.F * (b - c)

                # Crossover
                trial = np.copy(self.population[i])
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial[cross_points] = mutant[cross_points]
                trial = np.clip(trial, [b[0] for b in self.bounds], [b[1] for b in self.bounds])

                # Selection
                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    new_population.append(trial)
                else:
                    new_population.append(self.population[i])

            self.population = np.array(new_population)

        return best_solution, best_fitness, fitness_history

# Base Station - Multi-UAV - Multi-User Collaborative Environment
class DynamicCommunicationEnv:
    def __init__(self, num_bs=3, num_uav=3, num_users=5, terrain_size=1000, max_bs_deploy=12):
        self.terrain_size = terrain_size
        self.terrain_gen = TerrainGenerator(size=terrain_size, height_range=(100, 800), num_mountains=4)
        self.terrain = self.terrain_gen.terrain

        self.fc = 2.4e9
        self.c = 3e8
        self.L0 = 20 * np.log10(4 * np.pi / (self.c / self.fc))
        self.n1 = 3.4  # Base station path loss exponent
        self.n2 = 3.2  # UAV path loss exponent
        self.sigma1 = 10  # Base station shadow fading standard deviation
        self.sigma2 = 6  # UAV shadow fading standard deviation
        self.noise_power = 10 ** (-144 / 10)

        self.bs_snr_threshold = 10
        self.uav_snr_threshold = 10
        self.max_bs_deploy = max_bs_deploy
        # Generate 400 random candidate base station positions
        self.bs_candidates = self._generate_bs_candidates(400)
        # Select fixed base station positions
        self.fixed_bs_positions = random.sample(self.bs_candidates, 3)

        self.bs_list = self._init_base_stations(num_bs)
        self.deployed_bs_indices = []

        # Generate random UAV initial positions
        self.uav_start_positions = self._generate_uav_start_positions(num_uav)
        self.trajectory_colors = ['blue', 'green', 'purple']

        self.max_uav_height = 1000

        self.uav_list = self._init_uavs(num_uav)
        self.users = self._init_users(num_users)
        self.user_movement_range = 5

        self.final_user_snrs = []
        self.all_user_snrs = []
        self.universal_coverage_threshold = 10
        self.user_coverage_threshold = 0.8

        # Environmental complexity metrics
        self.esc = self.calculate_esc()
        self.evi_history = []

        # Coverage blind spot timer
        self.uncovered_timer = 0
        self.uncovered_threshold = 5  # Blind spot duration threshold

    # Calculate Environmental Static Complexity ESC
    def calculate_esc(self):
        h_values = self.terrain.flatten()
        h_mean = np.mean(h_values)
        esc = np.sqrt(np.mean((h_values - h_mean) ** 2))
        return esc

    # Calculate Environmental Volatility Index EVI
    def calculate_evi(self):
        if len(self.users) == 0:
            return 0
        user_mobility = 0
        for user in self.users:
            if len(user['trajectory']) > 1:
                p_prev = user['trajectory'][-2]
                p_curr = user['trajectory'][-1]
                user_mobility += np.linalg.norm(p_curr[:2] - p_prev[:2])
        user_mobility /= len(self.users)

        snr_values = [self.get_user_snr(user) for user in self.users]
        snr_std = np.std(snr_values) if snr_values else 0

        evi = 0.7 * user_mobility + 0.3 * snr_std
        self.evi_history.append(evi)
        return evi

    def _generate_bs_candidates(self, num_candidates):
        candidates = []
        for _ in range(num_candidates):
            x = np.random.uniform(0, self.terrain_size)
            y = np.random.uniform(0, self.terrain_size)
            z = self.terrain_gen.get_elevation(x, y) + 10  # 10 meters above terrain
            candidates.append(np.array([x, y, z]))
        return candidates

    def _generate_uav_start_positions(self, num_uav):
        start_positions = []
        center_radius = 0.3 * self.terrain_size  # Center region radius (30% of terrain size)
        center_x, center_y = self.terrain_size * 0.5, self.terrain_size * 0.5

        for _ in range(num_uav):
            while True:
                x = np.random.uniform(0, self.terrain_size)
                y = np.random.uniform(0, self.terrain_size)
                dist_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                if dist_from_center > center_radius:
                    break
            terrain_h = self.terrain_gen.get_elevation(x, y)
            z = terrain_h + np.random.uniform(50, 200)
            start_positions.append(np.array([x, y, z]))
        return start_positions

    def _calculate_coverage_radius(self, tx_power, is_uav=True):
        tx_power_dbm = 10 * np.log10(tx_power * 1000)
        noise_power_dbm = 10 * np.log10(self.noise_power * 1000)
        if is_uav:
            required_rx_power_dbm = noise_power_dbm + self.uav_snr_threshold
        else:
            required_rx_power_dbm = noise_power_dbm + self.bs_snr_threshold

        max_path_loss = tx_power_dbm - required_rx_power_dbm

        n = self.n2 if is_uav else self.n1
        freq_ghz = self.fc / 1e9
        FSPL_1m = 20 * np.log10(freq_ghz) + 32.44

        d_log = (max_path_loss - FSPL_1m) / (10 * n)
        radius = 10 ** d_log

        if is_uav:
            radius *= 0.9
        return max(radius, 1)

    def _init_base_stations(self, num_bs):
        bs_list = []
        for i, pos in enumerate(self.fixed_bs_positions[:num_bs]):
            x, y, z = pos
            power = np.random.uniform(0.01, 0.03)
            coverage_radius = self._calculate_coverage_radius(power, is_uav=False)

            bs = {
                'position': np.array([x, y, z]),
                'power': power,
                'frequency': self.fc,
                'coverage_radius': coverage_radius,
                'is_fixed': True
            }
            bs_list.append(bs)

        remaining_bs = max(0, num_bs - len(self.fixed_bs_positions))
        for _ in range(remaining_bs):
            candidate = random.choice(self.bs_candidates)
            power = np.random.uniform(0.01, 0.03)
            coverage_radius = self._calculate_coverage_radius(power, is_uav=False)

            bs = {
                'position': candidate,
                'power': power,
                'frequency': self.fc,
                'coverage_radius': coverage_radius,
                'is_fixed': False
            }
            bs_list.append(bs)

        return bs_list

    def _init_uavs(self, num_uav):
        uav_list = []
        for i in range(num_uav):
            start_x, start_y, start_z = self.uav_start_positions[i]
            power = np.random.uniform(0.005, 0.02)
            coverage_radius = self._calculate_coverage_radius(power, is_uav=True)

            uav = {
                'position': np.array([start_x, start_y, start_z]),
                'power': power,
                'velocity': np.zeros(3),
                'max_speed': 10,
                'max_v_speed': 5,
                'energy': 1000,
                'energy_consumption': 0,
                'trajectory': [np.array([start_x, start_y, start_z])],
                'coverage_radius': coverage_radius,
                'max_height': self.max_uav_height
            }
            uav_list.append(uav)
        return uav_list

    def _init_users(self, num_users):
        users = []
        for i in range(num_users):
            x = np.random.uniform(0, self.terrain_size)
            y = np.random.uniform(0, self.terrain_size)
            z = self.terrain_gen.get_elevation(x, y)

            priority = np.random.randint(1, 4)
            if priority == 1:
                snr_threshold = 12
            elif priority == 2:
                snr_threshold = 18
            else:
                snr_threshold = 24

            user = {
                'position': np.array([x, y, z]),
                'snr_threshold': snr_threshold,
                'priority': priority,
                'coverage_history': [],  # Record coverage history
                'trajectory': [np.array([x, y, z])]  # Add trajectory attribute, initial position as first point
            }
            users.append(user)
        return users

    def step_dynamic(self):
        for user in self.users:
            dx = np.random.uniform(-self.user_movement_range, self.user_movement_range)
            dy = np.random.uniform(-self.user_movement_range, self.user_movement_range)
            user['position'][0] = np.clip(user['position'][0] + dx, 0, self.terrain_size)
            user['position'][1] = np.clip(user['position'][1] + dy, 0, self.terrain_size)
            user['position'][2] = self.terrain_gen.get_elevation(user['position'][0], user['position'][1])
            user['trajectory'].append(user['position'].copy())
            user['coverage_history'].append(self.get_user_snr(user) >= user['snr_threshold'])

        for uav in self.uav_list:
            flight_energy = 0.3 * np.linalg.norm(uav['velocity'])
            comm_energy = 0.15 * uav['power']
            uav['energy'] -= (flight_energy + comm_energy)
            uav['energy_consumption'] += (flight_energy + comm_energy)

    # Calculate path loss
    def calculate_path_loss(self, tx_pos, rx_pos, is_uav=True):
        d = np.linalg.norm(tx_pos - rx_pos)
        d = max(d, 1)
        if is_uav:
            return self.L0 + 10 * self.n2 * np.log10(d) + np.random.normal(0, self.sigma2)
        else:
            return self.L0 + 10 * self.n1 * np.log10(d) + np.random.normal(0, self.sigma1)

    # Calculate user SNR
    def get_user_snr(self, user):
        total_power = 0
        for bs in self.bs_list:
            loss = self.calculate_path_loss(bs['position'], user['position'], is_uav=False)
            rx_power = 10 * np.log10(bs['power']) - loss
            total_power += 10 ** (rx_power / 10)
        for uav in self.uav_list:
            loss = self.calculate_path_loss(uav['position'], user['position'], is_uav=True)
            rx_power = 10 * np.log10(uav['power']) - loss
            total_power += 10 ** (rx_power / 10)
        return 10 * np.log10(total_power / self.noise_power)

    # Calculate SNR at other positions
    def get_snr_at_position(self, pos):
        total_power = 0
        for bs in self.bs_list:
            loss = self.calculate_path_loss(bs['position'], pos, is_uav=False)
            rx_power = 10 * np.log10(bs['power']) - loss
            total_power += 10 ** (rx_power / 10)
        for uav in self.uav_list:
            loss = self.calculate_path_loss(uav['position'], pos, is_uav=True)
            rx_power = 10 * np.log10(uav['power']) - loss
            total_power += 10 ** (rx_power / 10)
        return 10 * np.log10(total_power / self.noise_power)

    def is_position_covered(self, pos):
        for user in self.users:
            user_pos = user['position']
            if np.linalg.norm(pos[:2] - user_pos[:2]) < 1:
                return self.get_snr_at_position(pos) >= user['snr_threshold']
        return self.get_snr_at_position(pos) >= self.universal_coverage_threshold

    def get_state(self):
        state = []

        for uav in self.uav_list:
            state.extend(uav['position'] / self.terrain_size)
            state.extend(uav['velocity'] / uav['max_speed'])
            state.append(uav['energy'] / 1000)

        for user in self.users:
            state.extend(user['position'] / self.terrain_size)
            snr = self.get_user_snr(user)
            state.append(snr / 50)
            state.append(1 if snr >= user['snr_threshold'] else 0)

        bs_state = np.zeros(self.max_bs_deploy * 3)
        for i, bs_idx in enumerate(self.deployed_bs_indices):
            if i < self.max_bs_deploy:
                bs_state[i * 3:(i + 1) * 3] = self.bs_candidates[bs_idx] / self.terrain_size

        state.extend(bs_state)
        state.append(len(self.deployed_bs_indices) / self.max_bs_deploy)

        state.append(self.calculate_evi())  #
        state.append(self.esc / 100)

        return np.array(state, dtype=np.float32)

    # MTS-COM Short-term optimization objective function
    def short_term_objective(self, uav_positions):
        total_penalty = 0
        for user in self.users:
            snr = self.get_user_snr(user)
            if snr < user['snr_threshold']:

                min_dist = min(np.linalg.norm(user['position'][:2] - uav_pos[:2])
                               for uav_pos in uav_positions)
                total_penalty += min_dist

        movement_penalty = 0
        for i, uav in enumerate(self.uav_list):
            if len(uav['trajectory']) > 1:
                prev_pos = uav['trajectory'][-2]
                curr_pos = uav_positions[i]
                movement_penalty += np.linalg.norm(curr_pos - prev_pos)

        return total_penalty + 0.1 * movement_penalty

    # MTS-COM Medium-term optimization objective function
    def mid_term_objective(self, bs_positions):
        coverage_score = 0
        for user in self.users:
            min_dist = min(np.linalg.norm(user['position'] - bs_pos) for bs_pos in bs_positions)
            coverage_score += min_dist

        dispersion = 0
        if len(bs_positions) > 1:
            positions_2d = [pos[:2] for pos in bs_positions]
            dispersion = np.std([np.linalg.norm(p1 - p2) for p1 in positions_2d for p2 in positions_2d])

        deployment_cost = len(bs_positions) * 0.5

        return coverage_score + 0.3 * dispersion + deployment_cost

    def check_mid_term_trigger(self):
        uncovered_count = 0
        for user in self.users:
            if not user['coverage_history'] or not user['coverage_history'][-1]:
                uncovered_count += 1

        uncovered_ratio = uncovered_count / len(self.users) if self.users else 0

        if uncovered_ratio > 0.3:
            self.uncovered_timer += 1
        else:
            self.uncovered_timer = 0

        return self.uncovered_timer >= self.uncovered_threshold

    def take_action(self, actions, w_td3=0.5, w_de=0.5):
        uav_actions = actions[:3 * len(self.uav_list)]
        bs_actions = actions[3 * len(self.uav_list):]

        for i, uav in enumerate(self.uav_list):
            action = uav_actions[i * 3:(i + 1) * 3]
            uav['velocity'][0] = np.clip(action[0], -uav['max_speed'], uav['max_speed'])
            uav['velocity'][1] = np.clip(action[1], -uav['max_speed'], uav['max_speed'])
            uav['velocity'][2] = np.clip(action[2], -uav['max_v_speed'], uav['max_v_speed'])

            new_pos = uav['position'] + 20*uav['velocity']
            new_pos[0] = np.clip(new_pos[0], 0, self.terrain_size)
            new_pos[1] = np.clip(new_pos[1], 0, self.terrain_size)

            terrain_h = self.terrain_gen.get_elevation(new_pos[0], new_pos[1])
            min_height = terrain_h + 5
            new_pos[2] = np.clip(new_pos[2], min_height, uav['max_height'])

            uav['position'] = new_pos
            uav['trajectory'].append(uav['position'].copy())

        if len(self.deployed_bs_indices) < self.max_bs_deploy:
            deploy_decision = bs_actions[0]
            candidate_idx = int((bs_actions[1] + 1) / 2 * len(self.bs_candidates))
            candidate_idx = max(0, min(candidate_idx, len(self.bs_candidates) - 1))

            if deploy_decision > 0 and candidate_idx not in self.deployed_bs_indices:
                new_bs_pos = self.bs_candidates[candidate_idx]
                power = np.random.uniform(0.01, 0.03)
                coverage_radius = self._calculate_coverage_radius(power, is_uav=False)

                new_bs = {
                    'position': new_bs_pos,
                    'power': power,
                    'frequency': self.fc,
                    'coverage_radius': coverage_radius,
                    'is_fixed': False
                }
                self.bs_list.append(new_bs)
                self.deployed_bs_indices.append(candidate_idx)

        self.step_dynamic()
        current_snrs = [self.get_user_snr(user) for user in self.users]
        self.all_user_snrs.append(current_snrs)

        reward = self._calculate_reward()

        done = any(uav['energy'] <= 0 for uav in self.uav_list) or len(self.deployed_bs_indices) >= self.max_bs_deploy

        self.final_user_snrs = current_snrs

        return self.get_state(), reward, done

    # Calculate reward function
    def _calculate_reward(self):

        weighted_coverage = 0
        total_weight = 0
        uncovered_users = []
        for user in self.users:
            weight = user['priority']
            total_weight += weight
            snr = self.get_user_snr(user)
            is_covered = snr >= user['snr_threshold']
            weighted_coverage += weight * (1 if is_covered else 0)
            if not is_covered:
                uncovered_users.append(user)

        weighted_coverage_rate = weighted_coverage / total_weight if total_weight > 0 else 0
        volume_coverage = self.calculate_volume_coverage()
        volume_coverage_norm = volume_coverage / 100
        bs_spread_reward = self._calculate_bs_spread_reward()
        coverage_uniformity = self._calculate_coverage_uniformity()
        total_energy = sum(uav['energy_consumption'] for uav in self.uav_list)
        overlap_penalty = self._calculate_overlap_penalty()
        bs_deployment_cost = len(self.deployed_bs_indices) * 0.3
        user_proximity_reward = 0
        for uav in self.uav_list:
            uav_pos = uav['position'][:2]
            if uncovered_users:
                min_dist = min(np.linalg.norm(uav_pos - user['position'][:2]) for user in uncovered_users)
                proximity_reward = max(0, 1 - min_dist / uav['coverage_radius']) * 2
                user_proximity_reward += proximity_reward
            else:
                avg_dist = np.mean([np.linalg.norm(uav_pos - user['position'][:2]) for user in self.users])
                user_proximity_reward += max(0, 1 - avg_dist / (self.terrain_size * 0.5)) * 1
        uav_coverage_reward = 0
        for uav in self.uav_list:
            covered_by_uav = 0
            for user in self.users:
                uav_snr = self.get_single_snr(uav, user['position'], is_uav=True)
                if uav_snr >= user['snr_threshold']:
                    covered_by_uav += user['priority']
            uav_coverage_reward += covered_by_uav / len(self.users)

        if weighted_coverage_rate < self.user_coverage_threshold:
            reward = (12 * weighted_coverage_rate +
                      3 * volume_coverage_norm +
                      3 * bs_spread_reward +
                      2 * coverage_uniformity +
                      5 * user_proximity_reward +
                      4 * uav_coverage_reward +
                      -0.001 * total_energy -
                      3 * overlap_penalty -
                      bs_deployment_cost)
        else:
            reward = (6 * weighted_coverage_rate +
                      8 * volume_coverage_norm +
                      4 * bs_spread_reward +
                      4 * coverage_uniformity +
                      3 * user_proximity_reward +
                      2 * uav_coverage_reward +
                      -0.001 * total_energy -
                      4 * overlap_penalty -
                      bs_deployment_cost)

        return reward

    def get_single_snr(self, tx, pos, is_uav):
        loss = self.calculate_path_loss(tx['position'], pos, is_uav)
        rx_power = 10 * np.log10(tx['power']) - loss
        rx_power_w = 10 ** (rx_power / 10) / 1000
        return 10 * np.log10(rx_power_w / self.noise_power)

    def _calculate_bs_spread_reward(self):
        if len(self.bs_list) < 2:
            return 0.5
        total_distance = 0
        pair_count = 0
        positions = [bs['position'][:2] for bs in self.bs_list]
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                total_distance += np.linalg.norm(positions[i] - positions[j])
                pair_count += 1
        avg_distance = total_distance / pair_count if pair_count > 0 else 0
        max_possible_distance = np.sqrt(self.terrain_size ** 2 + self.terrain_size ** 2)
        return min(avg_distance / max_possible_distance, 1.0)

    def _calculate_coverage_uniformity(self):
        sample_points = 30
        snrs = []
        for _ in range(sample_points):
            x = np.random.uniform(0, self.terrain_size)
            y = np.random.uniform(0, self.terrain_size)
            z = self.terrain_gen.get_elevation(x, y)
            pos = np.array([x, y, z])
            is_user_area = any(np.linalg.norm(pos[:2] - user['position'][:2]) < 5 for user in self.users)
            if not is_user_area:
                snrs.append(self.get_snr_at_position(pos))
        if len(snrs) < 5:
            return 0.5
        snr_std = np.std(snrs)
        return max(1.0 - min(snr_std / 30, 1.0), 0.0)

    # Calculate signal overlap penalty
    def _calculate_overlap_penalty(self):
        sample_points = 30
        overlap_count = 0
        for _ in range(sample_points):
            x = np.random.uniform(0, self.terrain_size)
            y = np.random.uniform(0, self.terrain_size)
            z = self.terrain_gen.get_elevation(x, y)
            pos = np.array([x, y, z])
            coverage_count = 0
            for bs in self.bs_list:
                if np.linalg.norm(pos - bs['position']) <= bs['coverage_radius']:
                    coverage_count += 1
            for uav in self.uav_list:
                if np.linalg.norm(pos - uav['position']) <= uav['coverage_radius']:
                    coverage_count += 1
            if coverage_count > 1:
                overlap_count += 1
        return overlap_count / sample_points

    # Calculate user coverage rate
    def get_coverage_rate(self):
        if not self.final_user_snrs:
            self.final_user_snrs = [self.get_user_snr(user) for user in self.users]
        covered_count = sum(1 for snr, user in zip(self.final_user_snrs, self.users)
                            if snr >= user['snr_threshold'])
        return covered_count / len(self.users) * 100 if self.users else 0

    # Calculate spatial coverage rate
    def calculate_volume_coverage(self, vertical_layers=30, horizontal_resolution=10):
        grid_size = int(self.terrain_size / horizontal_resolution)
        cell_area = horizontal_resolution * horizontal_resolution
        total_volume = 0.0
        covered_volume = 0.0

        for i in range(grid_size):
            for j in range(grid_size):
                x = (i + 0.5) * horizontal_resolution
                y = (j + 0.5) * horizontal_resolution
                terrain_height = self.terrain_gen.get_elevation(x, y)
                total_volume += cell_area * 300
                for layer in range(vertical_layers):
                    height = terrain_height + (layer * 300 / vertical_layers)
                    pos = np.array([x, y, height])
                    if self.get_snr_at_position(pos) >= 10:
                        covered_volume += cell_area * (300 / vertical_layers)

        return (covered_volume / total_volume) * 100 if total_volume > 0 else 0

    def _save_users_data(self, writer):
        user_data = []
        for user_idx, user in enumerate(self.users):
            base_info = {
                "User ID": user_idx,
                "Priority": user['priority'],
                "SNR Threshold": user['snr_threshold'],
                "Initial X Coordinate": user['trajectory'][0][0] if user['trajectory'] else user['position'][0],
                "Initial Y Coordinate": user['trajectory'][0][1] if user['trajectory'] else user['position'][1],
                "Initial Z Coordinate": user['trajectory'][0][2] if user['trajectory'] else user['position'][2]
            }
            user_data.append(base_info)

            trajectory_data = []
            for time_step, pos in enumerate(user.get('trajectory', [])):
                is_covered = self.get_user_snr({
                    'position': pos,
                    'snr_threshold': user['snr_threshold']
                }) >= user['snr_threshold']

                trajectory_data.append({
                    "Time Step": time_step,
                    "X Coordinate": pos[0],
                    "Y Coordinate": pos[1],
                    "Z Coordinate": pos[2],
                })

            if not trajectory_data:
                trajectory_data.append({
                    "Time Step": 0,
                    "X Coordinate": user['position'][0],
                    "Y Coordinate": user['position'][1],
                    "Z Coordinate": user['position'][2],
                })

            pd.DataFrame(trajectory_data).to_excel(
                writer,
                sheet_name=f"User{user_idx}_Trajectory",
                index=False
            )

        pd.DataFrame(user_data).to_excel(writer, sheet_name="User Basic Information", index=False)

    def _save_uavs_data(self, writer):
        uav_data = []
        for uav_idx, uav in enumerate(self.uav_list):
            base_info = {
                "UAV ID": uav_idx,
                "Power": uav['power'],
                "Max Speed": uav['max_speed'],
                "Initial Energy": 1000,
                "Current Energy": uav['energy'],
                "Coverage Radius": uav['coverage_radius'],
                "Initial X Coordinate": uav['position'][0],
                "Initial Y Coordinate": uav['position'][1],
                "Initial Z Coordinate": uav['position'][2]
            }

            trajectory_data = []
            for time_step, pos in enumerate(uav['trajectory']):
                trajectory_data.append({
                    "Time Step": time_step,
                    "X Coordinate": pos[0],
                    "Y Coordinate": pos[1],
                    "Z Coordinate": pos[2],
                    "Energy": uav['energy'] if time_step == len(uav['trajectory']) - 1 else None
                })


            pd.DataFrame(trajectory_data).to_excel(writer, sheet_name=f"UAV{uav_idx}_Trajectory", index=False)
            uav_data.append(base_info)
        pd.DataFrame(uav_data).to_excel(writer, sheet_name="UAV Basic Information", index=False)

    def _save_terrain_data(self, writer):
        terrain_df = pd.DataFrame(self.terrain)
        terrain_df.columns = [f"X_{col}" for col in terrain_df.columns]
        terrain_df.index = [f"Y_{idx}" for idx in terrain_df.index]

        sample_step = max(1, self.terrain_size // 100)
        sampled_terrain = self.terrain[::sample_step, ::sample_step]
        sampled_df = pd.DataFrame(sampled_terrain)
        sampled_df.columns = [f"X_{col * sample_step}" for col in sampled_df.columns]
        sampled_df.index = [f"Y_{idx * sample_step}" for idx in sampled_df.index]

        sampled_df.to_excel(writer, sheet_name="Terrain Elevation Data (Sampled)")

        terrain_params = {
            "Parameter": ["Size", "Height Range (Min)", "Height Range (Max)", "Number of Mountains"],
            "Value": [self.terrain_size, self.terrain_gen.height_range[0],
                   self.terrain_gen.height_range[1], self.terrain_gen.num_mountains]
        }
        pd.DataFrame(terrain_params).to_excel(writer, sheet_name="Terrain Parameters", index=False)

    def save_data_to_excel(self, filename="simulation_data.xlsx"):
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            self._save_users_data(writer)
            self._save_uavs_data(writer)
            self._save_terrain_data(writer)
            print(f"Data successfully saved to {filename}")


# TD3 Algorithm Implementation
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = torch.relu(self.fc1(state))
        a = torch.relu(self.fc2(a))
        a = torch.tanh(self.fc3(a)) * self.max_action
        return a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1 Architecture
        self.q1_fc1 = nn.Linear(state_dim + action_dim, 256)
        self.q1_fc2 = nn.Linear(256, 256)
        self.q1_fc3 = nn.Linear(256, 1)

        # Q2 Architecture
        self.q2_fc1 = nn.Linear(state_dim + action_dim, 256)
        self.q2_fc2 = nn.Linear(256, 256)
        self.q2_fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = torch.relu(self.q1_fc1(sa))
        q1 = torch.relu(self.q1_fc2(q1))
        q1 = self.q1_fc3(q1)

        q2 = torch.relu(self.q2_fc1(sa))
        q2 = torch.relu(self.q2_fc2(q2))
        q2 = self.q2_fc3(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = torch.relu(self.q1_fc1(sa))
        q1 = torch.relu(self.q1_fc2(q1))
        q1 = self.q1_fc3(q1)
        return q1


class TD3Agent:
    def __init__(self, state_dim, action_dim, max_action, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5,
                 policy_freq=2):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0
        self.memory = deque(maxlen=1000000)
        self.expert_buffer = deque(maxlen=50000)  # Expert experience buffer
        self.optim_gain_threshold = 0.1

    def select_action(self, state, noisy=True):
        state = torch.FloatTensor(state.reshape(1, -1))
        action = self.actor(state).detach().numpy().flatten()
        if noisy:
            noise = np.random.normal(0, 0.1, size=action.shape)
            action = (action + noise).clip(-self.max_action, self.max_action)
        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # Add experience to expert experience buffer
    def add_expert_experience(self, state, action, optim_gain):
        if optim_gain > self.optim_gain_threshold:
            self.expert_buffer.append((state, action, optim_gain))

    # Sample from both regular buffer and expert buffer
    def sample_batch(self, batch_size):

        regular_batch_size = int(batch_size * 0.7)
        expert_batch_size = batch_size - regular_batch_size

        if len(self.memory) < regular_batch_size:
            regular_batch = list(self.memory)
        else:
            regular_batch = random.sample(self.memory, regular_batch_size)

        if len(self.expert_buffer) > 0 and expert_batch_size > 0:
            expert_experiences = list(self.expert_buffer)
            gains = np.array([exp[2] for exp in expert_experiences])
            gains = gains - np.min(gains) + 1e-5
            probs = gains / np.sum(gains)

            if len(expert_experiences) < expert_batch_size:
                expert_batch = expert_experiences
            else:
                indices = np.random.choice(len(expert_experiences), expert_batch_size, p=probs, replace=False)
                expert_batch = [expert_experiences[i] for i in indices]
        else:
            expert_batch = []

        batch = regular_batch + expert_batch
        random.shuffle(batch)

        state_batch = torch.FloatTensor([transition[0] for transition in batch])
        action_batch = torch.FloatTensor([transition[1] for transition in batch])
        reward_batch = torch.FloatTensor([transition[2] for transition in batch]).unsqueeze(1)
        next_state_batch = torch.FloatTensor([transition[3] for transition in batch])
        done_batch = torch.FloatTensor([transition[4] for transition in batch]).unsqueeze(1)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def train(self, batch_size=256):
        if len(self.memory) < batch_size:
            return
        self.total_it += 1

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.sample_batch(batch_size)

        with torch.no_grad():
            noise = torch.FloatTensor(action_batch.shape).data.normal_(0, self.policy_noise)
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state_batch) + noise).clamp(-self.max_action, self.max_action)
            target_Q1, target_Q2 = self.critic_target(next_state_batch, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward_batch + (1 - done_batch) * self.discount * target_Q

        current_Q1, current_Q2 = self.critic(state_batch, action_batch)
        critic_loss = nn.MSELoss()(current_Q1, target_Q) + nn.MSELoss()(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.Q1(state_batch, self.actor(state_batch)).mean()
            expert_loss = 0
            if len(self.expert_buffer) > 0:
                expert_states, expert_actions, _ = zip(
                    *random.sample(self.expert_buffer, min(32, len(self.expert_buffer))))
                expert_states = torch.FloatTensor(expert_states)
                expert_actions = torch.FloatTensor(expert_actions)
                current_actions = self.actor(expert_states)
                expert_loss = nn.MSELoss()(current_actions, expert_actions)

            total_actor_loss = actor_loss + 0.1 * expert_loss

            self.actor_optimizer.zero_grad()
            total_actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))


# Dynamic Weight Adaptive Mechanism (DWAM)
def update_weights(env, w_td3_base=0.5, w_de_base=0.5, w_min=0.1):
    evi = env.calculate_evi()
    w_td3 = w_td3_base + (1 - w_td3_base) * evi
    w_de = max(w_min, w_de_base * (1 - evi))
    # Normalize
    total = w_td3 + w_de
    w_td3 /= total
    w_de /= total
    return w_td3, w_de

# Short-term optimization
def de_short_term_optimize(env, a_td3, max_iter=20):
    num_uav = len(env.uav_list)
    uav_actions = a_td3[:3 * num_uav]

    bounds = []
    for i in range(num_uav):
        for j in range(3):
            current_val = uav_actions[i * 3 + j]
            bounds.append((current_val - 0.2, current_val + 0.2))  # Perturbation range of ±0.2

    # Create DE optimizer
    de = DEAlgorithm(pop_size=20, bounds=bounds, F=0.5, CR=0.7)

    def objective_func(x):
        uav_positions = []
        for i in range(num_uav):
            dx, dy, dz = x[i * 3:(i + 1) * 3]
            uav_pos = env.uav_list[i]['position'] + np.array([dx, dy, dz])
            uav_pos[0] = np.clip(uav_pos[0], 0, env.terrain_size)
            uav_pos[1] = np.clip(uav_pos[1], 0, env.terrain_size)
            terrain_h = env.terrain_gen.get_elevation(uav_pos[0], uav_pos[1])
            uav_pos[2] = np.clip(uav_pos[2], terrain_h + 5, env.uav_list[i]['max_height'])
            uav_positions.append(uav_pos)

        return env.short_term_objective(uav_positions)

    best_solution, best_fitness, _ = de.optimize(objective_func, max_iter=max_iter)
    return best_solution


def de_mid_term_optimize(env, max_iter=10):

    num_bs_to_deploy = 13
    bounds = []
    for _ in range(num_bs_to_deploy):
        bounds.append((0, env.terrain_size))
        bounds.append((0, env.terrain_size))
        bounds.append((100, 500))

    de = DEAlgorithm(pop_size=30, bounds=bounds, F=0.5, CR=0.7)

    def objective_func(x):
        bs_positions = [bs['position'] for bs in env.bs_list]
        for idx in x:
            idx = int(round(idx))
            if 0 <= idx < len(env.bs_candidates):
                bs_positions.append(env.bs_candidates[idx])

        return env.mid_term_objective(bs_positions)

    best_solution, best_fitness, _ = de.optimize(objective_func, max_iter=max_iter)
    return best_solution


def train_td3_agent(env, episodes=2):
    state_dim = env.get_state().shape[0]
    action_dim = 3 * len(env.uav_list) + 2
    max_action = 1.0

    agent = TD3Agent(state_dim, action_dim, max_action)

    rewards_history = []
    coverage_history = []
    energy_history = []
    volume_coverage_history = []
    weight_history = []

    training_log = []
    start_time = time.time()

    for episode in range(episodes):
        state = env.get_state()
        episode_reward = 0
        done = False
        # Calculate dynamic weights
        w_td3, w_de = update_weights(env)
        weight_history.append((w_td3, w_de))
        # TD3 generates initial action
        a_td3 = agent.select_action(state)
        # Short-term optimization (DE fine-tuning)
        a_de_short = de_short_term_optimize(env, a_td3)
        a_de_mid = np.zeros(2)
        if env.check_mid_term_trigger():
            mid_term_solution = de_mid_term_optimize(env)
            a_de_mid[0] = 1
            a_de_mid[1] = (mid_term_solution[0] / len(env.bs_candidates)) * 2 - 1
        a_final = np.concatenate([
            w_td3 * a_td3[:3 * len(env.uav_list)] + w_de * a_de_short,
            a_td3[3 * len(env.uav_list):]
        ])

        next_state, reward, done = env.take_action(a_final, w_td3, w_de)
        # Calculate optimization gain for BIEC
        init_perf = env.short_term_objective([uav['position'] for uav in env.uav_list])
        final_perf = env.short_term_objective([uav['position'] for uav in env.uav_list])
        optim_gain = (init_perf - final_perf) / (abs(init_perf) + 1e-5)

        # Store experience
        agent.remember(state, a_final, reward, next_state, done)
        # Store expert experience
        if optim_gain > agent.optim_gain_threshold:
            agent.add_expert_experience(state, a_final, optim_gain)
        agent.train()
        state = next_state
        episode_reward += reward

        rewards_history.append(episode_reward)
        yh = env.get_coverage_rate()
        coverage_history.append(yh)
        energy_history.append(sum(uav['energy_consumption'] for uav in env.uav_list))
        volume_coverage_val = env.calculate_volume_coverage()
        volume_coverage_history.append(volume_coverage_val)

        bs_deployed = len(env.deployed_bs_indices)
        training_log.append({
            'episode': episode + 1,
            'volume_coverage': volume_coverage_val,
            'Deployed Base Stations': bs_deployed,
            'Cumulative Energy Consumption': sum(uav['energy_consumption'] for uav in env.uav_list),
            'Episode Reward': episode_reward,
            'Time Elapsed (seconds)': time.time() - start_time
        })

        print(f"Episode {episode + 1}/{episodes}, "
              f"Volume Coverage: {volume_coverage_val:.2f}% ")

        if (episode + 1) % 20 == 0:
            agent.save(f"td3_agent_episode_{episode + 1}")

    total_time = time.time() - start_time
    print(f"Total runtime: {total_time:.2f} seconds")
    training_log.append({
        'episode': 'Total',
        'volume_coverage': None,
        'Deployed Base Stations': None,
        'Cumulative Energy Consumption': None,
        'Episode Reward': sum(rewards_history),
        'Time Elapsed (seconds)': total_time
    })

    df_log = pd.DataFrame(training_log)
    excel_filename = f'training_log_{time.strftime("%Y%m%d_%H%M%S")}.xlsx'
    df_log.to_excel(excel_filename, index=False)
    print(f"Training log saved to: {excel_filename}")

    return agent, rewards_history, coverage_history, volume_coverage_history, energy_history

def evaluate_agent(env, agent, visualize=True):
    state = env.get_state()
    done = False
    steps = 0

    coverage_rate = env.get_coverage_rate()
    volume_coverage = env.calculate_volume_coverage()
    bs_deployed = len(env.deployed_bs_indices)
    total_energy = sum(uav['energy_consumption'] for uav in env.uav_list)

    high_priority_users = [u for u in env.users if u['priority'] == 3]
    medium_priority_users = [u for u in env.users if u['priority'] == 2]
    low_priority_users = [u for u in env.users if u['priority'] == 1]

    high_coverage = sum(1 for u in high_priority_users if env.get_user_snr(u) >= u['snr_threshold']) / len(
        high_priority_users) * 100 if high_priority_users else 0
    medium_coverage = sum(1 for u in medium_priority_users if env.get_user_snr(u) >= u['snr_threshold']) / len(
        medium_priority_users) * 100 if medium_priority_users else 0
    low_coverage = sum(1 for u in low_priority_users if env.get_user_snr(u) >= u['snr_threshold']) / len(
        low_priority_users) * 100 if low_priority_users else 0

    eval_results = {
        'Total Steps': [steps],
        'Volume Coverage Rate (%)': [volume_coverage],
        'Deployed Base Stations': [bs_deployed],
        'Total Energy Consumption': [total_energy]
    }

    for i, bs in enumerate(env.bs_list):
        if bs['is_fixed']:
            eval_results[f'Fixed Base Station {i + 1} Position'] = [str(bs['position'])]
        else:
            eval_results[f'Deployed Base Station {i + 1} Position'] = [str(bs['position'])]

    for i, uav in enumerate(env.uav_list):
        eval_results[f'UAV {i + 1} Final Position'] = [str(uav['position'])]

    df_eval = pd.DataFrame(eval_results)
    eval_filename = f'evaluation_results_{time.strftime("%Y%m%d_%H%M%S")}.xlsx'
    df_eval.to_excel(eval_filename, index=False)
    print(f"Evaluation results saved to: {eval_filename}")

    print(f"\n===== Evaluation Results =====")
    print(f"Volume Coverage Rate: {volume_coverage:.2f}%")
    print(f"Deployed Base Stations: {bs_deployed}")
    print(f"Total Energy Consumption: {total_energy:.2f}")

    return coverage_rate, volume_coverage


if __name__ == "__main__":
    for i in range(20):
        env = DynamicCommunicationEnv()
        start_time = time.time()

        # Train model
        train_model = True
        if train_model:
            agent, rewards, coverage, volume_coverage, energy = train_td3_agent(env)
            agent.save("td3_agent_final")

        else:
            state_dim = env.get_state().shape[0]
            action_dim = 3 * len(env.uav_list) + 2
            max_action = 1.0
            agent = TD3Agent(state_dim, action_dim, max_action)
            agent.load("td3_agent_final")

        # Evaluate model
        coverage_rate, volume_coverage = evaluate_agent(env, agent)
        excel_name = f"Communication_Simulation_Data_{i + 1}.xlsx"
        env.save_data_to_excel(excel_name)