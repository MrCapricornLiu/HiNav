''' Batched navigation environment '''
import MatterSim
import json
import numpy as np
import math
import random
import networkx as nx
from collections import defaultdict
import os

from utils.data import load_nav_graphs, new_simulator

from vln.eval_utils import cal_dtw, cal_cls
from vln.data_utils import load_obj2vps

ERROR_MARGIN = 3.0


class EnvBatch(object):
    ''' A simple wrapper for a batch of MatterSim environments,
        using discretized viewpoints and pretrained features '''

    def __init__(self, connectivity_dir, scan_data_dir=None, feat_db=None, batch_size=100):
        """
        1. Load pretrained image feature
        2. Init the Simulator.
        :param feat_db: The name of file stored the feature.
        :param batch_size:  Used to create the simulator list.
        """
        self.feat_db = feat_db
        self.image_w = 640
        self.image_h = 480
        self.vfov = 60
        
        self.sims = []
        for i in range(batch_size):
            sim = MatterSim.Simulator()
            if scan_data_dir:
                sim.setDatasetPath(scan_data_dir)
            sim.setNavGraphPath(connectivity_dir)
            sim.setRenderingEnabled(False)
            sim.setDiscretizedViewingAngles(True)   # Set increment/decrement to 30 degree. (otherwise by radians)
            sim.setCameraResolution(self.image_w, self.image_h)
            sim.setCameraVFOV(math.radians(self.vfov))
            sim.setBatchSize(1)
            sim.initialize()
            self.sims.append(sim)

    def _make_id(self, scanId, viewpointId):
        return scanId + '_' + viewpointId

    def newEpisodes(self, scanIds, viewpointIds, headings):
        for i, (scanId, viewpointId, heading) in enumerate(zip(scanIds, viewpointIds, headings)):
            self.sims[i].newEpisode([scanId], [viewpointId], [heading], [0])

    def getStates(self):
        """
        Get list of states augmented with precomputed image features. rgb field will be empty.
        Agent's current view [0-35] (set only when viewing angles are discretized)
            [0-11] looking down, [12-23] looking at horizon, [24-35] looking up
        :return: [ ((36, 2048), sim_state) ] * batch_size
        """
        # feature_states = []
        states = []
        for i, sim in enumerate(self.sims):
            state = sim.getState()[0]
            states.append(state)
        return states

    def makeActions(self, actions):
        ''' Take an action using the full state dependent action interface (with batched input).
            Every action element should be an (index, heading, elevation) tuple. '''
        for i, (index, heading, elevation) in enumerate(actions):
            self.sims[i].makeAction([index], [heading], [elevation])


class R2RNavBatch(object):
    def __init__(
        self, instr_data, connectivity_dir, view_db=None,
        batch_size=64, seed=0, name=None, sel_data_idxs=None, args=None
    ):
        self.env = EnvBatch(connectivity_dir, feat_db=view_db, batch_size=batch_size,
                            scan_data_dir=args.scan_data_dir,  # for visualization
                            )
        self.data = instr_data
        self.scans = set([x['scan'] for x in self.data])
        self.connectivity_dir = connectivity_dir
        self.batch_size = batch_size
        self.name = name
        self.args = args

        self.gt_trajs = self._get_r2r_gt_trajs(self.data) # for evaluation

        # in validation, we would split the data
        if sel_data_idxs is not None:
            t_split, n_splits = sel_data_idxs
            ndata_per_split = len(self.data) // n_splits 
            start_idx = ndata_per_split * t_split
            if t_split == n_splits - 1:
                end_idx = None
            else:
                end_idx = start_idx + ndata_per_split
            self.data = self.data[start_idx: end_idx]

        self.seed = seed

        self.ix = 0
        self._load_nav_graphs()

        self.sim = new_simulator(self.connectivity_dir)

        self.buffered_state_dict = {}
        print('%s loaded with %d instructions, using splits: %s' % (
            self.__class__.__name__, len(self.data), self.name))

    def _get_r2r_gt_trajs(self, data):
        gt_trajs = {
            x['instr_id']: (x['scan'], x['path']) \
                for x in data if len(x['path']) > 1
        }
        return gt_trajs

    def size(self):
        return len(self.data)

    def _load_nav_graphs(self):
        """
        load graph from self.scan,
        Store the graph {scan_id: graph} in self.graphs
        Store the shortest path {scan_id: {view_id_x: {view_id_y: [path]} } } in self.paths
        Store the distances in self.distances. (Structure see above)
        Load connectivity graph for each scan, useful for reasoning about shortest paths
        :return: None
        """
        print('Loading navigation graphs for %d scans' % len(self.scans))
        self.graphs = load_nav_graphs(self.connectivity_dir, self.scans)
        self.shortest_paths = {}
        for scan, G in self.graphs.items():  # compute all shortest paths
            self.shortest_paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
        self.shortest_distances = {}
        for scan, G in self.graphs.items():  # compute all shortest paths
            self.shortest_distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def _next_minibatch(self, batch_size=None, **kwargs):
        """
        Store the minibach in 'self.batch'
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        batch = self.data[self.ix: self.ix+batch_size]
        if len(batch) < batch_size:
            random.shuffle(self.data)
            self.ix = batch_size - len(batch)
            batch += self.data[:self.ix]
        else:
            self.ix += batch_size
        self.batch = batch

    def reset_epoch(self, shuffle=False):
        ''' Reset the data index to beginning of epoch. Primarily for testing.
            You must still call reset() for a new episode. '''
        if shuffle:
            random.shuffle(self.data)
        self.ix = 0

    # 很重要，优化的部分
    def make_candidate(self, current_state):
        def _loc_distance(loc):
            return np.sqrt(loc.rel_heading ** 2 + loc.rel_elevation ** 2)

        # --- 从传入的 state 获取信息 ---
        scanId = current_state.scanId
        viewpointId = current_state.location.viewpointId
        viewId = current_state.viewIndex # 当前内部视角索引

        # --- 计算当前相对角度基准 (用于旧逻辑) ---
        base_heading_current = (viewId % 12) * math.radians(30)
        base_elevation_current = (viewId // 12 - 1) * math.radians(30)

        adj_dict = {}
        long_id = f"{scanId}_{viewpointId}" # 缓存 key

        if long_id not in self.buffered_state_dict:
            # --- 内部探索逻辑 (保持不变) ---
            state_after_exploration = None
            for ix in range(36):
                if ix == 0:
                    self.sim.newEpisode([scanId], [viewpointId], [0], [math.radians(-30)])
                elif ix % 12 == 0:
                    self.sim.makeAction([0], [1.0], [1.0])
                else:
                    self.sim.makeAction([0], [1.0], [0])

                state = self.sim.getState()[0]
                state_after_exploration = state
                assert state.viewIndex == ix

                # 计算相对角度 (相对于探索时的相机朝向)
                heading = state.heading - base_heading_current
                elevation = state.elevation - base_elevation_current

                for j, loc in enumerate(state.navigableLocations[1:]):
                    distance = _loc_distance(loc)
                    # 计算候选点相对于探索时相机的相对角度
                    loc_heading = heading + loc.rel_heading
                    loc_elevation = elevation + loc.rel_elevation

                    if (loc.viewpointId not in adj_dict or
                            distance < adj_dict[loc.viewpointId]['distance']):

                        blip2_caption = None
                        # 初始 image 路径 (会被覆盖，但先记录)
                        initial_img_path = os.path.join(self.args.img_root, scanId, viewpointId, str(ix) + '.jpg')

                        adj_dict[loc.viewpointId] = {
                            'heading': loc_heading, # 相对角度
                            'elevation': loc_elevation, # 相对角度
                            "normalized_heading": state.heading + loc.rel_heading, # 绝对角度
                            "normalized_elevation": state.elevation + loc.rel_elevation, # 绝对角度
                            'scanId': scanId,
                            'viewpointId': loc.viewpointId,
                            'pointId': ix, # 发现时的视角
                            'distance': distance,
                            'idx': j + 1,
                            'position': (loc.x, loc.y, loc.z),
                            'caption': blip2_caption,
                            'image': initial_img_path, # 临时路径
                            'absolute_heading': state.heading,   # 发现时相机的绝对朝向
                            'absolute_elevation': state.elevation, # 发现时相机的绝对仰角
                        }
            # --- 内部探索结束 ---

            candidate = list(adj_dict.values())

            # --- 新增：优化候选点代表性视图 ---
            if candidate:
                try:
                    # 使用传入的 current_state 获取调用时的位置信息
                    current_pos = np.array([current_state.location.x, current_state.location.y, current_state.location.z])
                except AttributeError:
                    print("Error: current_state object missing location attributes. Returning original candidates.")
                    # 出错时，确保恢复 image 路径为原始发现时的路径
                    for cand in candidate:
                         cand['image'] = os.path.join(self.args.img_root, scanId, viewpointId, str(cand['pointId']) + '.jpg')
                    return candidate # 返回未优化的列表

                optimized_candidates = []
                for i_cand, cand in enumerate(candidate):
                    cand_copy = cand.copy() # 操作副本

                    target_pos = np.array(cand['position'])
                    direction_vec = target_pos - current_pos

                    if np.linalg.norm(direction_vec) < 1e-6:
                        # 目标点与当前点重合，恢复原始路径并保留
                        cand_copy['image'] = os.path.join(self.args.img_root, scanId, viewpointId, str(cand['pointId']) + '.jpg')
                        optimized_candidates.append(cand_copy)
                        continue

                    # 1. 计算绝对世界方向角
                    dx, dy, dz = direction_vec[0], direction_vec[1], direction_vec[2]
                    horizontal_dist = np.sqrt(dx**2 + dy**2)
                    if horizontal_dist < 1e-6: horizontal_dist = 1e-6
                    target_elevation = np.arctan2(dz, horizontal_dist)
                    target_heading = np.arctan2(dx, dy) # 相对+Y轴

                    # 2. 寻找最接近的离散视角索引 (0-35)
                    best_ix = -1
                    min_angle_diff = float('inf')
                    for test_ix in range(36):
                        angle_per_step = math.radians(30)
                        level = test_ix // 12
                        step_in_level = test_ix % 12
                        base_elevation = (level - 1) * angle_per_step
                        base_heading = step_in_level * angle_per_step
                        base_heading = (base_heading + math.pi) % (2 * math.pi) - math.pi

                        heading_diff = target_heading - base_heading
                        heading_diff = (heading_diff + math.pi) % (2 * math.pi) - math.pi
                        elevation_diff = target_elevation - base_elevation
                        current_angle_diff = abs(heading_diff) + abs(elevation_diff)

                        if current_angle_diff < min_angle_diff:
                            min_angle_diff = current_angle_diff
                            best_ix = test_ix

                    # 3. 更新候选点信息
                    if best_ix != -1:
                        cand_copy['pointId'] = best_ix # 更新 pointId
                        # 使用 img_root 构建新路径
                        cand_copy['image'] = os.path.join(self.args.img_root, scanId, viewpointId, str(best_ix) + '.jpg')
                    else:
                        # 如果没找到最佳（理论上不应发生），恢复原始路径
                         cand_copy['image'] = os.path.join(self.args.img_root, scanId, viewpointId, str(cand['pointId']) + '.jpg')

                    optimized_candidates.append(cand_copy)

                candidate = optimized_candidates # 使用优化后的列表
            # --- 优化结束 ---

            # --- 更新缓存 ---
            # 直接存储优化后的 candidate 字典列表，确保所有键都被保留
            # （假设 candidate 列表中的字典包含了所有我们需要的键，包括优化后的 image/pointId 和原始的 absolute_heading/elevation 等）
            self.buffered_state_dict[long_id] = [c.copy() for c in candidate] # 存储副本列表
            # --- 缓存结束 ---

            return candidate # 返回优化后的

        else: # 从缓存读取
            candidate_from_buffer = self.buffered_state_dict[long_id]

            candidate_new = []
            # 重新计算相对于当前 viewId 的 heading/elevation
            base_heading_current = (current_state.viewIndex % 12) * math.radians(30)
            base_elevation_current = (current_state.viewIndex // 12 - 1) * math.radians(30)
            for i_cand_buf, c in enumerate(candidate_from_buffer):
                c_new = c.copy()
                # 使用缓存中保存的绝对角度计算相对角度
                # 确保 buffer 中保存了 normalized_heading/elevation
                if 'normalized_heading' in c_new and 'normalized_elevation' in c_new:
                    c_new['heading'] = c_new['normalized_heading'] - base_heading_current
                    c_new['elevation'] = c_new['normalized_elevation'] - base_elevation_current
                else:
                     # 如果缓存没有绝对角度，无法计算相对角度，可能需要报错或设为0
                     c_new['heading'] = 0
                     c_new['elevation'] = 0
                candidate_new.append(c_new)

            return candidate_new

    def _get_obs(self):
        obs = []
        for i, state in enumerate(self.env.getStates()):
            item = self.batch[i]

            candidate = self.make_candidate(state)

            if 'instr_encoding' in item.keys():
                instr_encoding = item['instr_encoding']
            else:
                instr_encoding = None

            ob = {
                'instr_id' : item['instr_id'],
                'scan' : state.scanId,
                'viewpoint' : state.location.viewpointId,
                'viewIndex' : state.viewIndex,
                'position': (state.location.x, state.location.y, state.location.z),
                'heading' : state.heading,
                'elevation' : state.elevation,
                'candidate': candidate,
                'navigableLocations' : state.navigableLocations,
                'instruction' : item['instruction'],
                'instr_encoding': instr_encoding, # item['instr_encoding'],
                'gt_path' : item['path'],
                'path_id' : item['path_id'],
                'surrounding_tags': None
            }
            # RL reward. The negative distance between the state and the final state
            # There are multiple gt end viewpoints on REVERIE. 
            if ob['instr_id'] in self.gt_trajs:
                ob['distance'] = self.shortest_distances[ob['scan']][ob['viewpoint']][item['path'][-1]]
            else:
                ob['distance'] = 0

            obs.append(ob)
        return obs

    def reset(self, **kwargs):
        ''' Load a new minibatch / episodes. '''
        self._next_minibatch(**kwargs)
        
        scanIds = [item['scan'] for item in self.batch]
        viewpointIds = [item['path'][0] for item in self.batch]
        headings = [item['heading'] for item in self.batch]
        self.env.newEpisodes(scanIds, viewpointIds, headings)
        return self._get_obs()

    def step(self, actions):
        ''' Take action (same interface as makeActions) '''
        self.env.makeActions(actions)
        return self._get_obs()

    ############### Nav Evaluation ###############
    def _get_nearest(self, shortest_distances, goal_id, path):
        near_id = path[0]
        near_d = shortest_distances[near_id][goal_id]
        for item in path:
            d = shortest_distances[item][goal_id]
            if d < near_d:
                near_id = item
                near_d = d
        return near_id

    def _eval_r2r_item(self, scan, pred_path, gt_path):
        scores = {}

        shortest_distances = self.shortest_distances[scan]

        path = sum(pred_path, [])
        assert gt_path[0] == path[0], 'Result trajectories should include the start position'

        nearest_position = self._get_nearest(shortest_distances, gt_path[-1], path)

        scores['nav_error'] = shortest_distances[path[-1]][gt_path[-1]]
        scores['oracle_error'] = shortest_distances[nearest_position][gt_path[-1]]

        scores['action_steps'] = len(pred_path) - 1
        scores['trajectory_steps'] = len(path) - 1
        scores['trajectory_lengths'] = np.sum([shortest_distances[a][b] for a, b in zip(path[:-1], path[1:])])

        gt_lengths = np.sum([shortest_distances[a][b] for a, b in zip(gt_path[:-1], gt_path[1:])])
        
        scores['success'] = float(scores['nav_error'] < ERROR_MARGIN)
        scores['spl'] = scores['success'] * gt_lengths / max(scores['trajectory_lengths'], gt_lengths, 0.01)
        scores['oracle_success'] = float(scores['oracle_error'] < ERROR_MARGIN)

        scores.update(
            cal_dtw(shortest_distances, path, gt_path, scores['success'], ERROR_MARGIN)
        )
        scores['CLS'] = cal_cls(shortest_distances, path, gt_path, ERROR_MARGIN)

        return scores

    def eval_metrics(self, preds, dataset):
        ''' Evaluate each r2r trajectory based on how close it got to the goal location
        the path contains [view_id, angle, vofv]'''
        print('eval %d predictions' % (len(preds)))

        metrics = defaultdict(list)
        for item in preds:
            instr_id = item['instr_id']
            traj = item['trajectory']

            scan, gt_traj = self.gt_trajs[instr_id]
            traj_scores = self._eval_r2r_item(scan, traj, gt_traj)

            for k, v in traj_scores.items():
                metrics[k].append(v)
            metrics['instr_id'].append(instr_id)

        avg_metrics = {
            # 'action_steps': np.mean(metrics['action_steps']),
            'steps': np.mean(metrics['trajectory_steps']),
            'lengths': np.mean(metrics['trajectory_lengths']),
            'nav_error': np.mean(metrics['nav_error']),
            'oracle_error': np.mean(metrics['oracle_error']),
            'sr': np.mean(metrics['success']) * 100,
            'oracle_sr': np.mean(metrics['oracle_success']) * 100,
            'spl': np.mean(metrics['spl']) * 100,
        }

        return avg_metrics, metrics

