import numpy as np

import pyximport; pyximport.install()
import partial_plates.partial_plate_gofs

from .partial_plate_gofs import calc_gofs
from .eulers import get_v_beta_from_euler, get_ve_vn_from_v_beta

def plate_vels(plate, lons, lats, poles_df=None):
    pole = poles_df[poles_df.plate == plate]
    v, beta = get_v_beta_from_euler(lon1=np.radians(pole.lon.values[0]),
                                    lat1=np.radians(pole.lat.values[0]),
                                    lon0=np.radians(lons),
                                    lat0=np.radians(lats),
                                    rotation_rate=np.radians(
                                               pole.rate_deg_Myr.values) / 1e6)
    
    plate_ve, plate_vn = get_ve_vn_from_v_beta(v, beta, return_mm=True)
    
    return plate_ve, plate_vn


def partial_plate_vels(plate, lons, lats, coeff=1., poles_df=None):
    return coeff * np.array(plate_vels(plate, lons, lats, poles_df))


def multiplate_vels(plates, lons, lats, coeffs, poles_df=None):
    if len(plates) != len(coeffs):
        raise Exception('need same number of plates and coeffs')
    
    vs = np.sum( partial_plate_vels(plate, lons, lats, coeffs[i])
                 for i, plate in enumerate(plates))
    ve, vn = vs
    return vs


def make_coeffs(n_iters, n_plates, eq_1=True):
    coeffs = np.random.random((n_iters, n_plates))
    coeffs = coeffs / coeffs.sum(axis=1)[:,np.newaxis]
        
    if eq_1 is True:
        pass
    else:
        coeffs = coeffs * np.random.random((n_iters,1))
    return coeffs


def calc_gof(pvs, coeff_row, ve, vn):
    pred_vels = pvs * coeff_row[:,np.newaxis]
    pred_e, pred_n = pred_vels.sum(axis=0)
    
    return np.sqrt( (ve-pred_e)**2 + (vn-pred_n)**2 )


def find_plate_coeffs(lon, lat, ve, vn, plates, n_iters, eq_1=True,
                      poles_df=None):
    n_plates = len(plates)
    
    coeff_priors = make_coeffs(n_iters, n_plates, eq_1=eq_1)
    
    pvs = np.squeeze(np.array([plate_vels(plate, lon, lat, poles_df) 
                               for plate in plates]))
    
    gofs = calc_gofs(pvs, coeff_priors, ve, vn)
    
    return coeff_priors[np.argmin(gofs), :]
