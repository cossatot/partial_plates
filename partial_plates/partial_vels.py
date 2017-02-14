import numpy as np

import pyximport; pyximport.install()
import partial_plates.partial_plate_gofs

from .partial_plate_gofs import calc_gofs


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


def find_plate_coeffs(lon, lat, ve, vn, plates, n_iters, eq_1=True):
    n_plates = len(plates)
    
    coeff_priors = make_coeffs(n_iters, n_plates, eq_1=eq_1)
    
    pvs = np.squeeze(np.array([plate_vels(plate, lon, lat)
                               for plate in plates]))
    gofs = calc_gofs(pvs, coeff_priors, ve, vn)
    
    return coeff_priors[np.argmin(gofs), :]
