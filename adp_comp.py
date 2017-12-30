import numpy as np
import kernel_regression as kr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared, WhiteKernel, ConstantKernel as C



def get_D_KR(trajs, pos, subsampling=1, dt=0.001):
    """
    Estimates the position-dependent diffusion constant using KDE method.
    Parameters
    ----------
    trajs : list of arrays
        List, where each element is a trajectory.  The trajectory is assumed to be a 2D array-like, where each row is a timestep and each column is a coordinate.  Note: each trajectory must have the same timesteps.
    pos: list
        Positions where you would like to estimate diffusion coefficient using the model
    subsampling: float, optional
        The subsampling used when calculate trajectories. Default is 1.
    dt: float, optional
         Timestep for the Bronwnian_dynamics.  Default is 0.001, which is a decent choice for a harmonic with force constant 1.

    Returns
    -------
    D_kr : array
        Estimate of the diffusion constant corresponding to pos.
    """
    kernel = kr.KernelRegression(kernel="rbf", gamma=np.logspace(-2, 2, 10))
    dim = np.size(trajs[0][0])
    print('KR Dimension is %d' %(dim))
    D_kr = np.zeros([dim,dim,len(pos)])
    for i in xrange(dim):
        for j in xrange(dim):
            y_trajs = np.zeros([len(trajs), len(trajs[0])-1])
            x_trajs = np.zeros([len(trajs), len(trajs[0])-1, np.size(trajs[0][0])])
            for m in xrange(len(trajs)):
                for n in xrange(len(trajs[0])-1):
                    y_trajs[m][n] = (trajs[m][n+1][i]-trajs[m][n][i])*(trajs[m][n+1][j]-trajs[m][n][j])
                    x_trajs[m][n] = trajs[m][n]

            x = x_trajs.reshape(len(trajs)*(len(trajs[0])-1),dim)
            y = y_trajs.flatten()
            model = kernel.fit(x,y)
            D_kr[i][j] = (model.predict(pos))/(dt*2*subsampling)

    return D_kr


def get_D_GPR(trajs, pos, subsampling=1, dt=0.001):
    """
    Estimates the position-dependent diffusion constant using GPR method.

    Parameters
    ----------
    trajs : list of arrays
        List, where each element is a trajectory.  The trajectory is assumed to be a 2D array-like, where each row is a timestep and each column is a coordinate.  Note: each trajectory must have the same timesteps.
    pos: list
        Positions where you would like to estimate diffusion coefficient using the model
    subsampling: float, optional
        The subsampling used when calculate trajectories. Default is 10.
    dt: float, optional
         Timestep for the Bronwnian_dynamics.  Default is 0.001, which is a decent choice for a harmonic with force constant 1.

    Returns
    -------
    D_gpr : array
        Estimate of the diffusion constant corresponding to pos.
    """

    dim = np.size(trajs[0][0])
    print('GPR Dimension is %d' %(dim))
    D_gpr = np.zeros([dim,dim,len(pos)])
    D_gpr_cov = np.zeros([dim, dim, len(pos), len(pos)])
    for i in xrange(dim):
        for j in xrange(dim):
            y_trajs = np.zeros([len(trajs), len(trajs[0])-1])
            x_trajs = np.zeros([len(trajs), len(trajs[0])-1, np.size(trajs[0][0])])
            for m in xrange(len(trajs)):
                for n in xrange(len(trajs[0])-1):
                    y_trajs[m][n] = (trajs[m][n+1][i]-trajs[m][n][i])*(trajs[m][n+1][j]-trajs[m][n][j])
                    x_trajs[m][n] = trajs[m][n]

            x = x_trajs.reshape(len(trajs)*(len(trajs[0])-1),dim)
            y = y_trajs.flatten()

            # Construct the GPR kernel
            kernel = C(1.0) * ExpSineSquared(length_scale = 1.0, periodicity = 2*np.pi) * RBF(length_scale=1.0) + WhiteKernel(noise_level=np.var(y), noise_level_bounds='fixed')
            # Fit the Gaussian process, predict D
            GP = GaussianProcessRegressor(alpha=0.0,kernel=kernel,normalize_y=True)
            # print 'x shape', np.shape(x)
            # print 'y shape', np.shape(y)
            GP.fit(x,y)
            D_gpr[i][j], D_gpr_cov[i][j]  = GP.predict(pos, return_cov=True)
            D_gpr[i][j]/=(dt*2*subsampling)

    return D_gpr

def find_neighbouring(i, p, N):
    """
    Find the neighbouring bin indexes around bin_i.

    Parameters
    ----------
    i : int
        index of the bin we would like to find the neighbours.
    p: int
        period, the repetition period.
    N: int
        the number of total bins.


    Returns
    -------
    neighbour_index : array
        indexes of neighbouring bins and bin_i as an array(totally 9 elements).
    """
    neighbour_index = []
    # if not vertex
    if (i != N - p) and (i != N - 1) and (i != 0) and (i != p - 1):
        # if at upper edge but not vertex
        if i >= N - p:
            print'upper edge'
            up = i % p
            upleft = (i-1) % p
            upright = (i+1) % p
            left = i - 1
            right = i + 1
            down = i - p
            downright = i - p + 1
            downleft = i - p - 1
        # if at lower edge but not vertex
        elif i < p:
            print'lower edge'
            up = i + p
            upleft = i + p - 1
            upright = i + p + 1
            left = i - 1
            right = i + 1
            down = N - (p - i)
            downright = N - (p - i - 1)
            downleft = N - (p - i + 1)
        # if at left edge but not vertex
        elif i % p == 0:
            print'left edge'
            up = i + p
            upleft = i + (2 * p - 1)
            upright = i + p + 1
            left = i + (p - 1)
            right = i + 1
            down = i - p
            downright = i - p + 1
            downleft = i - 1
        # if at right edge but not vertex
        elif i % p == 4:
            print'right edge'
            up = i + p
            upleft = i + p - 1
            upright = i + 1
            left = i - 1
            right = i - (p - 1)
            down = i - p
            downright = i - (2*p - 1)
            downleft = i - p - 1
        # if in the middle
        else:
            print'middle'
            up = i + p
            upleft = i + p - 1
            upright = i + p + 1
            left = i - 1
            right = i + 1
            down = i - p
            downright = i - p + 1
            downleft = i - p - 1


    #if vertex
    else:
        # upperleft vertex
        if (i >= (N - p)) and (i % p == 0):
            print'upleft vertex'
            up = i % p
            upleft = (i - 1) % p
            upright = (i + 1) % p
            left = i + (p - 1)
            right = i + 1
            down = i - p
            downright = i - p + 1
            downleft = i - 1
        # upperright vertex
        elif (i >=(N - p))and(i % p == 4):
            print'upright vertex'
            up = i % p
            upleft = (i - 1) % p
            upright = (i + 1) % p
            left = i - 1
            right = i - (p - 1)
            down = i - p
            downright = i - (2*p - 1)
            downleft = i - p - 1
        # lowerleft vertex
        elif (i < p)and(i % p == 0):
            print'lowerleft vertex'
            up = i + p
            upleft = i + (2*p - 1)
            upright = i + p + 1
            left = i + (p - 1)
            right = i + 1
            down = N - (p - i)
            downright = N - (p - (i + 1))
            downleft = N - 1

        # lowerright vertex
        elif (i < p)and(i % p == 4):
            print'lowerright vertex'
            up = i + p
            upleft = i + p - 1
            upright = i + 1
            left = i - 1
            right = i - (p - 1)
            down = N - (p - i)
            downright = N - p
            downleft = N - (p - (i - 1))

    # put all indexes into neighbour_index
    neighbour_index = np.array([up, upleft, upright, left, right, down, downright, downleft, i])

    return neighbour_index




def main():
    # Load in trajectories into trajs file.
    trajs = np.load('trajs.npy')
    num_umbrella = 20
    block_region = 1
    kT = 0.616033281788
    dt = 0.001
    subsampling = 10

    neighbouring_window_index = find_neighbouring(block_region-1, num_umbrella, num_umbrella**2)

    trajs = np.array([trajs[neighbouring_window_index[0]],trajs[neighbouring_window_index[1]],trajs[neighbouring_window_index[2]],trajs[neighbouring_window_index[3]],trajs[neighbouring_window_index[4]],trajs[neighbouring_window_index[5]],trajs[neighbouring_window_index[6]],trajs[neighbouring_window_index[7]],trajs[neighbouring_window_index[8]]])
    trajs = trajs[::subsampling]

    center = np.load('centers.npy')
    pos = center[block_region-1]

    print("KR estimation...")
    D_KR = get_D_KR(trajs,pos,subsampling,dt)
    print("GPR estimation...")
    D_GPR = get_D_GPR(trajs, pos, subsampling, dt)

    print'D_KR%d'%(block_region-1), D_KR
    print'D_GPR%d'%(block_region-1), D_GPR



    np.save('D_KR%d'%(block_region-1), D_KR)
    np.save('D_GPR%d'%(block_region-1), D_GPR)

    return



if __name__ == "__main__":
    main()
