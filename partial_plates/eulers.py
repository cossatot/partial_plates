#All code copyright 2014, Richard Styron.  All rights reserved.

import numpy as np

def get_aa(lat_pole=None, lat_site=None, lon_pole=None, lon_site=None):
    """Cosine distance formula"""
    
    aa =  np.arccos(np.sin(lat_site) * np.sin(lat_pole)
                     + np.cos(lat_pole) * np.cos(lat_site) 
                     * np.cos(lon_pole - lon_site) )
    return aa


def get_C(lat_pole=None, lon_pole=None, lon_site=None, lat_site=None):
    aa = get_aa(lat_pole=lat_pole, lon_pole=lon_pole, 
                lat_site=lat_site, lon_site=lon_site)

    C = np.arcsin(np.cos(lat_pole) * np.sin(lon_pole - lon_site)
                  / np.sin(aa))

    if np.isscalar(C):
        if lat_site > lat_pole:
            C = np.pi - C
    else:
        C[lat_site > lat_pole] = np.pi - C
    
    return C


def get_v(rotation_rate, aa, radius=6371000,
          return_mm = False):
    
    v = rotation_rate * radius * np.sin(aa)
    return v*1e3 if return_mm==True else v


def get_beta(C):
    return np.pi/2 + C


def get_ve_vn_from_v_beta(v, beta, return_mm = False):
    
    if return_mm == True:
        v *= 1e3
    vn = v * np.cos(beta)
    ve = v * np.sin(beta)
    
    return ve, vn


def get_v_beta_from_euler(lat_pole=None, lat_site=None, lon_pole=None, 
                          lon_site=None, rotation_rate=None):
    
    aa = get_aa(lat_pole=lat_pole, lat_site=lat_site, lon_pole=lon_pole,
              lon_site=lon_site)
    
    C = get_C(lat_pole=lat_pole, lon_pole=lon_pole, 
              lon_site=lon_site, lat_site=lat_site)
    
    v = get_v(rotation_rate, aa)
    
    beta = get_beta(C)
    
    return v, beta


def arc_rad_dir(lat_site=None, lat_pole=None, lon_site=None, lon_pole=None):
    
    return np.arctan( (lon_pole - lon_site) / (lat_pole - lat_site) )


def get_arc_strike(lat_site=None, lat_pole=None, lon_site=None, lon_pole=None):
    
    theta = arc_rad_dir(lat_site = lat_site, lat_pole = lat_pole, 
                        lon_site = lon_site, lon_pole = lon_pole)
    
    return theta + np.pi/2


def get_v_az(ve, vn):
    return np.arctan2(vn, ve) + np.pi/2


def arc_par_v(ve, vn, arc_strike):
    v_mag = np.sqrt(ve**2 + vn**2)
    v_az = get_v_az(ve, vn)
    
    return -v_mag * np.cos(v_az - arc_strike)
    
    
def get_arc_par_vels(lat_site=None, lat_pole=None, lon_site=None,
                     lon_pole=None, ve=None, vn=None, return_components=False):
    
    arc_strike = get_arc_strike(lat_site, lat_pole, lon_site, lon_pole)
    
    par_v = arc_par_v(ve, vn, arc_strike)

    if return_components==True:
        ve_par, vn_par = get_ve_vn_from_v_beta(par_v, arc_strike)
        return ve_par, vn_par
    else:
        return par_v


def angle_difference(angle1, angle2, return_abs = False, units = 'radians'):
    if units == 'degrees':
        angle1 = np.radians(angle1)
        angle2 = np.radians(angle2)

    if np.isscalar(angle1) and np.isscalar(angle2):
        diff = angle_difference_scalar(angle1, angle2)
    else:
        diff = angle_difference_vector(angle1, angle2)

    if units == 'degrees':
        angle1 = np.degrees(angle1)
        angle2 = np.degrees(angle2)

    return diff if return_abs == False else np.abs(diff)


def angle_difference_scalar(angle1, angle2):
    difference = angle2 - angle1
    while difference < - np.pi:
        difference += 2 * np.pi
    while difference > np.pi:
        difference -= 2 * np.pi
    return difference


def angle_difference_vector(angle1_vec, angle2_vec):
    angle1_vec = np.array(angle1_vec)
    angle2_vec = np.array(angle2_vec)
    difference = angle2_vec - angle1_vec
    difference[difference < -np.pi] += 2 *  np.pi
    difference[difference > np.pi] -= 2 * np.pi
    
    return difference


def add_poles(lon_cb=0., lat_cb=0., rot_cb=0.,
              lon_ba=0., lat_ba=0., rot_ba=0.,
              input_units='degrees', output_units='degrees'):
    '''
    Calculates the Euler pole and rotation rate for plate C relative to
    plate A based on the CB and BA rotation information.
    '''

    if input_units == 'degrees':
        lon_cb = np.radians(lon_cb)
        lat_cb = np.radians(lat_cb)
        rot_cb = np.radians(rot_cb)
        lon_ba = np.radians(lon_ba)
        lat_ba = np.radians(lat_ba)
        rot_ba = np.radians(rot_ba)
        
    #TODO: put check for zero rotation rates here

    if rot_cb == 0. and rot_ba != 0.:
        lon_ca, lat_ca, rot_ca = lon_ba, lat_ba, rot_ba

    elif rot_ba == 0. and rot_cb != 0.:
        lon_ca, lat_ca, rot_ca = lon_cb, lat_cb, rot_cb

    elif rot_ba == 0. and rot_cb == 0.:
        # consider raising ValueError
        lon_ca, lat_ca, rot_ca = 0., 0., 0.

    else:
        x_ca = (rot_cb * np.cos(lat_cb) * np.cos(lon_cb) 
               +rot_ba * np.cos(lat_ba) * np.cos(lon_ba))

        y_ca = (rot_cb * np.cos(lat_cb) * np.sin(lon_cb) 
               +rot_ba * np.cos(lat_ba) * np.sin(lon_ba))

        z_ca = rot_cb * np.sin(lat_cb) + rot_ba * np.sin(lat_ba)
        
        rot_ca = np.sqrt(x_ca**2 + y_ca**2 + z_ca**2)
        lat_ca = np.arcsin(z_ca / rot_ca)
        lon_ca = np.arctan2( y_ca , x_ca )
    
    if output_units == 'degrees':
        return np.degrees((lon_ca, lat_ca, rot_ca))
    else:
        return lon_ca, lat_ca, rot_ca
