
from shapely.geometry import LinearRing, Point, MultiPoint, Polygon

from env.vehicle import State
from env.map_base import Area
from configs import *
DEBUG = False
LEVEL_NORMAL = "Normal"
LEVEL_COMPLEX = "Complex"
LEVEL_EXTREM = "Extrem"

EXTREM_PARK_LOT_LENGTH = min(LENGTH*1.2, LENGTH+0.9)

def _get_surrounding_obstacle(dest:State, obstacles:list):
    dest_rb, dest_rf, dest_lf, dest_lb = list(dest.create_box().coords)[:-1]
    pt_dest_lc = Point(_get_midpoint(dest_lf, dest_lb))
    pt_dest_rc = Point(_get_midpoint(dest_rf, dest_rb))
    pt_dest_bc = Point(_get_midpoint(dest_lb, dest_rb))
    pt_dest_fc = Point(_get_midpoint(dest_lf, dest_rf))
    pts = [pt_dest_lc, pt_dest_rc, pt_dest_fc, pt_dest_bc]
    detected_obstacle = []
    for pt in pts:
        detected_obstacle.append(_get_nearest_obstacle(pt, obstacles, LENGTH/2, \
            no_consider_obsts=detected_obstacle))
    return detected_obstacle

def get_map_level(start:State, dest:State, obstacle_list:list):
    '''
    get the difficulty of a map.

    Return:
        map_level (str): "Normal", "Complex", or "Extrem"
    '''
    if len(obstacle_list) <= 1:
        return LEVEL_NORMAL
    if isinstance(obstacle_list[0], Area):
        obstacles = list([obs.shape for obs in obstacle_list])
    elif isinstance(obstacle_list[0], Polygon):
        obstacles = list([obs.exterior for obs in obstacle_list])
    elif not isinstance(obstacle_list[0], LinearRing):
        raise NotImplementedError('obstacle should be `Area`, `Polygon` or `LinearRing`!')

    if _check_extrem_level(start, dest, obstacles):
        return LEVEL_EXTREM
    
    distance_exceed = False
    if start.loc.distance(dest.loc) > MAX_DRIVE_DISTANCE:
        if DEBUG:
            print('distance:',start.loc.distance(dest.loc))
        distance_exceed = True

    obst_left, obst_right, obst_front, obst_back = _get_surrounding_obstacle(dest, obstacles)
    dest_rb, dest_rf, dest_lf, dest_lb = list(dest.create_box().coords)[:-1]
    # determind the parking case: bay, parallel
    if obst_left and obst_right and (obst_front is None): # bay oarking
        if distance_exceed or not _has_enough_space(dest, obstacles, width=MIN_PARK_LOT_WIDTH_DICT['Normal']):
            return LEVEL_COMPLEX
        # use shapely `MultiPoint.minimum_rotated_rectangle` method to get the minimum free space
        free_space_key_pts = []
        dest_heading = dest.heading
        free_space_key_pts.append(_pt_translate(dest_lf, dest_heading, 0.2))
        free_space_key_pts.append(_pt_translate(dest_rf, dest_heading, 0.2))
        free_space_key_pts.append(_pt_translate(dest_lf, dest_heading, BAY_PARK_WALL_DIST_DICT['Normal']-0.5))
        free_space_key_pts.append(_pt_translate(dest_rf, dest_heading, BAY_PARK_WALL_DIST_DICT['Normal']-0.5))
        # free_space_key_pts.extend(list(start.create_box().coords)[:-1])
        free_space_key_pts.append(start.loc)
        free_space = MultiPoint(free_space_key_pts).minimum_rotated_rectangle
        free_space_valid = True
        for obst in obstacles:
            if obst.equals(obst_left) or obst.equals(obst_right):
                continue
            elif free_space.intersects(obst):
                if DEBUG:
                    print('free space invlid', list(free_space.exterior.coords))
                free_space_valid = False
        return LEVEL_NORMAL if free_space_valid else LEVEL_COMPLEX

    elif obst_front and obst_back: # parallel parking
        if distance_exceed or not _has_enough_space(dest, obstacles, length=MIN_PARK_LOT_LEN_DICT['Normal']):
            return LEVEL_COMPLEX
        out_direction = dest.heading + np.pi/2
        if np.cos(out_direction)*(start.loc.x-dest.loc.x) + \
            np.sin(out_direction)*(start.loc.y-dest.loc.y) < 0:
            out_direction += np.pi
            key_pt_front = dest_rf
            key_pt_back = dest_rb
        else:
            key_pt_front = dest_lf
            key_pt_back = dest_lb
        free_space_key_pts = []
        free_space_key_pts.append(_pt_translate(key_pt_front, out_direction, 0.2))
        free_space_key_pts.append(_pt_translate(key_pt_back, out_direction, 0.2))
        free_space_key_pts.append(_pt_translate(key_pt_front, out_direction, PARA_PARK_WALL_DIST_DICT['Normal']-0.5))
        free_space_key_pts.append(_pt_translate(key_pt_back, out_direction, PARA_PARK_WALL_DIST_DICT['Normal']-0.5))
        free_space_key_pts.extend(list(start.create_box().coords)[:-1])
        free_space_key_pts.append(start.loc)
        free_space = MultiPoint(free_space_key_pts).minimum_rotated_rectangle
        free_space_valid = True
        for obst in obstacles:
            if obst.equals(obst_back) or obst.equals(obst_front):
                continue
            elif free_space.intersects(obst):
                if DEBUG:
                    print('free space invlid', list(free_space.exterior.coords))
                free_space_valid = False
        return LEVEL_NORMAL if free_space_valid else LEVEL_COMPLEX
    elif (obst_left is None or obst_right is None) and (obst_front is None or obst_back is None):
        return LEVEL_NORMAL
    else: # otherwise the parking case is not normal
        if DEBUG:
            print('unconsidered case',obst_front, obst_back, obst_left, obst_right)
        return LEVEL_COMPLEX

def _pt_translate(pt:tuple, heading:float, dist:float):
    x_off = np.cos(heading)*dist
    y_off = np.sin(heading)*dist
    return (pt[0]+x_off, pt[1]+y_off)
    

def _check_extrem_level(start:State, dest:State, obstacles:list):
    obst_left, obst_right, obst_front, obst_back = _get_surrounding_obstacle(dest, obstacles)
    # distance criterion
    if start.loc.distance(dest.loc) > 30.0:
        if obst_front and obst_back and \
            not _has_enough_space(dest, obstacles, length=MIN_PARK_LOT_LEN_DICT['Normal']):
            return True
        if obst_left and obst_right and \
            not _has_enough_space(dest, obstacles, width=MIN_PARK_LOT_WIDTH_DICT['Normal']):
            return True

    # narrow parallel parking lot
    if obst_front and obst_back and \
        not _has_enough_space(dest, obstacles, length=EXTREM_PARK_LOT_LENGTH):
        return True

    return False

def _get_midpoint(pt1:tuple, pt2:tuple): 
    return ((pt1[0]+pt2[0])/2 , (pt1[1]+pt2[1])/2)

def _get_nearest_obstacle(pt:Point, obatacles:list, max_min_dist=99., no_consider_obsts=[]):
    min_dist = max_min_dist
    nearest_obstacle = None
    for obst in obatacles:
        not_consider = False
        for obst2 in no_consider_obsts:
            if obst2 is not None and obst.equals(obst2):
                not_consider = True
                break
        if not_consider:
            continue
        dist = pt.distance(obst)
        if dist < min_dist:
            min_dist = dist
            nearest_obstacle = obst
    return nearest_obstacle

def _has_enough_space(pos:State, obstacles, width=None, length=None):
    'Check whether there is enough space around the given position.'
    dest_box = pos.create_box()
    width_qualified = True
    if width is not None:
        obst_left, obst_right, _, _ = _get_surrounding_obstacle(pos, obstacles)
        if obst_left is None or obst_right is None:
            width_qualified = True
        elif obst_left.distance(dest_box) + obst_right.distance(dest_box) + WIDTH < width:
            if DEBUG:
                print('no enough width:', obst_left.distance(obst_right))
            width_qualified = False
        else:
            width_qualified = True

    length_qualified = True
    if length is not None:
        _, _, obst_front, obst_back = _get_surrounding_obstacle(pos, obstacles)
        if obst_front is None or obst_back is None:
            length_qualified = True
        elif obst_front.distance(dest_box) + obst_back.distance(dest_box) + LENGTH < length:
            length_qualified = False
            if DEBUG:
                print('no enough length:', obst_front.distance(obst_back))
        else:
            length_qualified = True
    return width_qualified and length_qualified