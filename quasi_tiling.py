import numpy as np
## The geometry is defined here and the shape of the quasicrystal unit is defined here
# the unit consists of a square and a equilateral triangles on all its sides

## The tiling is achieved by combination of two threepoint generations, the first one
# uses three points of the square and the secon one uses two points of the square and 
# one additional point from the top of the equilateral triangle in direction of 
# the two points chosen from the square


def build_points2(p,a):
    """
    Parameters:
        p (numpy.ndarray) - three points used to generate
        a (int, float) - side length of the lattice
    """ 
    rr = np.array([[p[0,:]-p[1,:]],[p[0,:]-p[2,:]],[p[1,:]-p[2,:]]])
    r = np.array([np.linalg.norm(i) for i in rr])

    ## Check the validity of points in square formation
    long =  (1+np.sqrt(3))*a/2
    dia = np.sqrt(2)
    expected = np.array([a, a,np.sqrt(2)])
    if not np.allclose(np.sort(r), np.sort(expected)):
        return 0,0

 
    ## Find the middle point and vector between two side points
    peak = np.array([])
    hyp = np.array([0,0])
    midpoint = np.array([0,0])
    for n in range(3):
        if round(r[n],3) != a:
            peak = np.array(p[2-n,:])
            midpoint = np.array((p[2 if n != 0 else 0,:] + p[abs(1-n),:])/2) 
            hyp = rr[n]
    mp = (midpoint - peak)/np.linalg.norm(midpoint - peak)
    
    ## If the three points are not oriented in the positive direction the points are disregarded, the generation is proceding from the
    # center so there is no reason to generate backwards
    invd = np.dot(mp,np.array([1,1]))
    if invd<0:
        return 1,0

    p1 = peak + dia*mp

    short = np.sqrt(2)/2*a
    hp = hyp/np.linalg.norm(hyp)
    

    ## Define the rest of the points needed to finish the mesh
    p2 = midpoint - short*hp
    p5 = midpoint + short*hp
    
    yy = (peak - p5)/np.linalg.norm(peak-p5)
    xx = (peak - p2)/np.linalg.norm(peak-p2)
    
    p7 = midpoint + yy*long 
    p6 = midpoint - yy*long
    p4 = midpoint + xx*long
    p3 = midpoint - xx*long

    return (np.vstack((peak,p7,p2,p3,p1,p6,p5,p4)), peak)

def build_points(p,a,invert):
    """
    Parameters:
        p (numpy.ndarray) - three points used to generate
        a (int, float) - side length of the lattice
        invert (int) - 1 or -1 deciding the orientation of the geometry
    """
    ## Invert must be 1 or -1 for proper functioning, it is used for checking both possible orientations of the geometry
    # based on the three chosen points

    rr = np.array([[p[0,:]-p[1,:]],[p[0,:]-p[2,:]],[p[1,:]-p[2,:]]])
    r = np.array([np.linalg.norm(i) for i in rr])

    ## Check the validity of points in a shape described before
    long =  a * 2 * np.sin(np.radians(75))
    expected = np.array([a, a,long])
    if not np.allclose(np.sort(r), np.sort(expected)):
        return 0,0

 
    ## Find the middle point and vector between two side points
    peak = np.array([])
    hyp = np.array([0,0])
    midpoint = np.array([0,0])
    for n in range(3):
        if round(r[n],3) != a:
            peak = np.array(p[2-n,:])
            midpoint = np.array((p[2 if n != 0 else 0,:] + p[abs(1-n),:])/2) 
            hyp = rr[n]
    mp = (midpoint - peak)/np.linalg.norm(midpoint - peak)

    ## If the three points are not oriented in the positive direction the points are disregarded, the generation is proceding from the
    # center so there is no reason to generate backwards
    invd = np.dot(mp,np.array([1,1]))
    if invd<0:
        return 1,0


    p1 = peak + long*mp

    ## Other points based on the orientation decided by invert
    short = a*np.cos(np.radians(75))
    hp = invert * hyp/np.linalg.norm(hyp)
    a0 = (p1+peak)/2

    p2 = a0+hp*short
    p3 = a0-hp*(long-short)
    p0 = midpoint - invert * hyp/2 
    p4 = p0 + (p2-peak)/np.linalg.norm(p2-peak)*a
    p5 = p4 - mp*long
    p6 = p0 + hp*long

    return (np.vstack((peak,p5,p0,p3,p4,p1,p2,p6)), peak)

def rotated_points(points):
    """
    Rotates input points with respect to hexagonal symetry around the origin

    Parameters:
        points (numpy.ndarray) - input points
    """
    t = np.pi/3
    rr = points.shape[0]
    rot_mat = np.array([[np.cos(t),-np.sin(t)],[np.sin(t), np.cos(t)]])
    for i in range(6):
        for p in range(rr):
            points = np.vstack((points,np.linalg.matrix_power(rot_mat,i+1) @ points[p,:]))
    return points

def sort_by_distance_from_origin(points):
    """
    Sorts the input points by distance from origin

    Parameters:
        points (numpy.ndarray) - input points
    """
    return points[np.argsort(np.linalg.norm(points, axis=1))]

