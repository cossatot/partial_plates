import numpy as np


def okada_slip(fault_seg, station_coords):
    ok = fault_seg.to_okada()
    
    # need to make adjustment for the stupid Okada coordinate system
    # where y=0 is at the bottom edge of the fault.
    y_ok = ok['d'] / np.tan(ok['delta'])
    
    sx, sy = fault_oblique_merc(station_coords, fault_seg)
    fx, fy = fault_oblique_merc(ok['coords'], fault_seg)
    
    x0, y0 = fx[0], fy[0]
    
    sx0, sy0 = sx - x0, sy-y0
    
    syo = sy0 + y_ok
    
    dx, dy, dz = okada85(sx0, syo, ok['L'], ok['W'], ok['d'], ok['delta'], 
                         ok['U'])
    
    de, dn = rotate_vecs(dx, dy, fault_seg.strike)

    return de, dn, dz


def rotate_vecs(dx, dy, strike_rad):
    #ang = strike_to_angle(strike)
    ang = np.pi/2 - strike_rad
    
    de = dx * np.cos(ang) - dy * np.sin(ang)
    dn = dx * np.sin(ang) + dy * np.cos(ang)
    
    return de, dn


class FaultSeg(object):
    def __init__(self, lon0=0., lat0=0., lon1=0., lat1=0., 
                 dip=45., rake=90., strike=None, top_depth=0.,
                 bottom_depth=15., ss=0., ds=0., ts=0.):

        self.lat0 = lat0
        self.lon0 = lon0
        self.lat1 = lat1
        self.lon1 = lon1
        self.ss = ss
        self.ds = ds
        self.ts = ts
        self.dip = dip
        self.rake = rake
        self.top_depth = top_depth
        self.bottom_depth = bottom_depth
        self.lat0_r = np.radians(lat0)
        self.lon0_r = np.radians(lon0)
        self.lat1_r = np.radians(lat1)
        self.lon1_r = np.radians(lon1)
        self.dip_r = np.radians(dip)
        self.rake_r = np.radians(rake)
        self.length = gc_dist((self.lon0_r, self.lat0_r), 
                              (self.lon1_r, self.lat1_r))
        
        if strike == None:
            self.strike = calc_strike(self.lon0_r, self.lat0_r, 
                                      self.lon1_r, self.lat1_r)
        else:
            self.strike = strike
        
    def to_okada(self):
        
        okada_params = {'L':self.length,
                        'delta': self.dip_r,
                        'd' : self.bottom_depth,
                        'W' : (self.bottom_depth - self.top_depth) \
                               / np.sin(self.dip_r),
                        #'U' : U_from_rake_slip(rake),
                        'U' : (self.ss, self.ds, self.ts),
                        'coords' : np.array([[self.lon0, self.lat0],
                                             [self.lon1, self.lat1]])}
        return okada_params
        

def calc_strike(lon0, lat0, lon1, lat1):
    
    y = np.sin(lon1-lon0) * np.cos(lat1)
    x = np.cos(lat0) * np.sin(lat1) \
        - np.sin(lat0) * np.cos(lat1) * np.cos(lon1-lon0)
    
    az_r = np.arctan2(y,x)
    
    #angle = np.degrees(np.arctan2(y,x))
    
    #az = 90 - angle #-(angle - 90)
    
    #while az < 0:
    #    az += 360
    #while az > 360:
    #    az -= 360
    
    #return az
    return az_r


def gc_dist(p0, p1, R=6371, input='radians'):
    '''
    Returns the (approximate) great-circle distance
    between two points (in long-lat). Returned
    distance in kilometers.
     '''
    if input == 'degrees':
        p0 = np.radians(p0)
        p1 = np.radians(p1)
    
    x = (p0[0]-p1[0]) * np.cos((p0[1]+p1[1]) /2)
    y = p0[1] - p1[1]
    
    return R * np.sqrt((x**2 + y**2))


def rotate_vecs(dx, dy, strike):
    #ang = strike_to_angle(strike)
    ang = np.pi/2 - strike
    
    de = dx * np.cos(ang) - dy * np.sin(ang)
    dn = dx * np.sin(ang) + dy * np.cos(ang)
    
    return de, dn


def sind(angle):
    return np.sin(np.radians(angle))


def cosd(angle):
    return np.cos(np.radians(angle))


def atand(arg):
    return np.degrees(np.arctan(arg))


def tand(angle):
    return np.tan(np.radians(angle))


def fault_oblique_merc(station_coords, fault_seg, R=6371.):
    '''
    Carries out oblique Mercator projection of the data contained in arrays
    `lon` and `lat`, such that the x-axis is parallel to a fault trace and the
    y-axis is perpendicular. The fault trace is defined using endpoint coordinates
    lon1, lat1, lon2, lat2.
    
    From Meade et al., Blocks, 'faultobliquemerc.m'.
    '''
    #todo: make it work for scalar coords
    lon = station_coords[:,0]
    lat = station_coords[:,1]
    
    sdata = lon.shape
    
    lon1 = np.ones(sdata) * fault_seg.lon0
    lat1 = np.ones(sdata) * fault_seg.lat0
    lon2 = np.ones(sdata) * fault_seg.lon1
    lat2 = np.ones(sdata) * fault_seg.lat1
    
    #trig functions
    clat = cosd(lat)
    slat = sind(lat)
    clat1 = cosd(lat1)
    slat1 = sind(lat1)
    clat2 = cosd(lat2)
    slat2 = sind(lat2)
    clon1 = cosd(lon1)
    slon1 = sind(lon1)
    clon2 = cosd(lon2)
    slon2 = sind(lon2)
    
    # Pole longitude
    num = clat1 * slat2 * clon1 - slat1 * clat2 * clon2
    den = slat1 * clat2 * slon2 - clat1 * slat2 * slon1
    lonp = np.degrees(np.arctan2(num, den))
    
    # Pole latitude
    latp = atand(-cosd(lonp - lon1) / tand(lat1))
    sp = np.sign(latp)
    # Choose northern hemisphere pole
    lonp[latp < 0] += 180.
    latp[latp < 0] *= -1.
    # Find origin longitude
    lon0 = lonp + 90.
    lon0[lon0 > 180] -= 360.
    
    clatp = cosd(latp)
    slatp = sind(latp)
    dlon = lon - lon0
    A = slatp * slat - clatp * clat * sind(dlon)
    
    x = np.arctan((tand(lat) * clatp + slatp * sind(dlon)) / cosd(dlon))
    x = x - (cosd(dlon) > 0) * np.pi + np.pi/2
    y = np.arctanh(A)
    x = -sp * x * R
    y = -sp * y * R
    
    return x, y


def okada85(x, y, L, W, d, delta, U, tol=1e-10):
    '''
    Description:
      computes displacements at points (x,y) for a fault with
      width W, length L, and depth d.  The fault has one end on the
      origin and the other end at (x=L,y=0).  Slip on the fault is
      described by U.

    Arguments:
      x: along strike coordinate of output locations (can be a vector)
      y: perpendicular to strike coordinate of output locations (can be a 
         vector)
      L: length of fault
      W: width of fault
      d: depth of the bottom of the fault (a fault which ruptures the surface
         with have d=W, and d<W will give absurd results)
      delta: fault dip.  0<delta<pi/2.0 will dip in the -y direction. and
             pi/2<delta<pi will dip in the +y direction... i think.
      U: a three components vector with strike-slip, dip-slip, and tensile
         components of slip

    output:
      tuple where each components is a vector of displacement in the x, y, or z  
      direction

    Written by Trever Hines.
    '''
    mu = 3.2e10
    lamb = 3.2e10
    x = np.array(x)
    y = np.array(y)
    cosdel = np.cos(delta)
    sindel = np.sin(delta)
    p = y*cosdel + d*sindel
    q = y*sindel - d*cosdel
    r = np.sqrt(np.power(y,2.0) +
                np.power(x,2.0) +
                np.power(d,2.0))

    def f(eps,eta):
        y_til = eta*cosdel + q*sindel
        d_til = eta*sindel - q*cosdel
        R = np.sqrt(np.power(eps,2.0) +
                    np.power(eta,2.0) +
                    np.power(q,2.0))
        X = np.sqrt(np.power(eps,2.0) +
                    np.power(q,2.0))
  
        if cosdel < tol:
          I1 = (-mu/(2*(lamb + mu))*
                eps*q/np.power((R + d_til),2.0))
          I3 = (mu/(2*(lamb + mu))*
                (eta/(R + d_til) +
                 y_til*q/np.power(R + d_til,2.0) -
                 np.log(R + eta)))
          I2 = (mu/(lamb + mu)*
                -np.log(R + eta) - I3)
          I4 = (-mu/(lamb + mu)*
                q/(R + d_til))
          I5 = (-mu/(lamb + mu)*
                eps*sindel/(R + d_til))
        else:
          I5 = (mu/(lamb + mu)*
                (2.0/cosdel)*
                np.arctan((eta*(X + q*cosdel) + X*(R + X)*sindel)/
                          (eps*(R + X)*cosdel)))
          I4 = (mu/(lamb + mu)*
                1.0/cosdel*
                (np.log(R + d_til) - sindel*np.log(R + eta)))
          I3 = (mu/(lamb + mu)*
                (y_til/(cosdel*(R+d_til)) - np.log(R + eta)) +
                sindel/cosdel*I4)
          I2 = (mu/(lamb + mu)*
                -np.log(R + eta) - I3)
          I1 = (mu/(lamb + mu)*
                (-eps/(cosdel*(R + d_til))) -
                sindel/cosdel*I5)
  
        u1 = (-U[0]/(2*np.pi)*
              (eps*q/(R*(R + eta)) +
               np.arctan(eps*eta/(q*R)) +
               I1*sindel))
        u2 = (-U[0]/(2*np.pi)*
              (y_til*q/(R*(R + eta)) +
               q*cosdel/(R + eta) +
               I2*sindel))
        u3 = (-U[0]/(2*np.pi)*
              (d_til*q/(R*(R + eta)) +
               q*sindel/(R + eta) +
               I4*sindel))
  
        u1 += (-U[1]/(2*np.pi)*
               (q/R - I3*sindel*cosdel))
        u2 += (-U[1]/(2*np.pi)*
               (y_til*q/(R*(R + eps)) +
                cosdel*np.arctan(eps*eta/(q*R)) -
                I1*sindel*cosdel))
        u3 += (-U[1]/(2*np.pi)*
               (d_til*q/(R*(R + eps)) +
                sindel*np.arctan(eps*eta/(q*R)) -
                I5*sindel*cosdel))
  
        u1 += (U[2]/(2*np.pi)*
               (np.power(q,2.0)/(R*(R + eta)) -
                I3 * np.power(sindel,2.0)))
        u2 += (U[2]/(2*np.pi)*
               (-d_til*q/(R*(R + eps)) -
                sindel*(eps*q/(R*(R + eta)) - np.arctan(eps*eta/(q*R))) -
                I1*np.power(sindel,2.0)))
        u3 += (U[2]/(2*np.pi)*
               (y_til*q/(R*(R + eps)) +
                cosdel*(eps*q/(R*(R + eta)) - np.arctan(eps*eta/(q*R))) -
                I5*np.power(sindel,2.0)))
        return (u1,u2,u3)
  
    disp1 = f(x,p)[0] - f(x,p - W)[0] - f(x - L,p)[0] + f(x-L,p-W)[0]
    disp2 = f(x,p)[1] - f(x,p - W)[1] - f(x - L,p)[1] + f(x-L,p-W)[1]
    disp3 = f(x,p)[2] - f(x,p - W)[2] - f(x - L,p)[2] + f(x-L,p-W)[2]

    return (disp1,disp2,disp3)


