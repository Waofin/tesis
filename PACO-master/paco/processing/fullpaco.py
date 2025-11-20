"""
This module implements ALGORITHM 1 from Flasseur et al.
It uses patch covariance to determine the signal to noise
ratio of a signal within ADI image data.

OPTIMIZED VERSION: Includes parallelization using joblib with threading backend
"""
from paco.util.util import *
from .paco import PACO
import matplotlib.pyplot as plt
import sys
try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    import warnings
    warnings.warn("joblib not available, parallelization disabled")

class FullPACO(PACO):
    """
    Algorithm Functions
    """     
    def PACOCalc(self, phi0s, cpu = 1):
        """
        PACO_calc
        
        This function iterates of a list of test points (phi0) and a list
        of angles between frames to produce 'a' and b', which can be used to
        generate a signal to noise map where SNR = b/sqrt(a) at each pixel.
        
        OPTIMIZED VERSION: Supports parallelization using joblib with threading backend
        
        phi0s : int arr
            Array of pixel locations to estimate companion position
        params: dict
            Dictionary of parameters about the psf, containing either the width
            of a gaussian distribution, or a label 'psf_template'
        scale : float
            Resolution scaling
        model_name: str
            Name of the template for the off-axis PSF
        cpu : int >= 1
            Number of cores to use for parallel processing. Uses joblib with threading backend.
        """  
        npx = len(phi0s)  # Number of pixels in an image
        dim = self.m_width/2
        a = np.zeros(npx) # Setup output arrays
        b = np.zeros(npx)
        T = len(self.m_im_stack) # Number of temporal frames
        mask = createCircularMask(self.m_psf.shape,radius = self.m_psf_rad)
        if not hasattr(self, 'm_p_size') or self.m_p_size != len(mask[mask]):
            self.m_p_size = len(mask[mask])      

        print("Running PACO...")
        if cpu > 1 and JOBLIB_AVAILABLE:
            print(f"  Using parallel processing with {cpu} cores (threading backend)")
        
        # Set up coordinates so 0 is at the center of the image                     
        x, y = np.meshgrid(np.arange(-dim, dim), np.arange(-dim, dim))
        
        # OPTIMIZATION: Parallelize the main loop if cpu > 1 and joblib is available
        if cpu > 1 and JOBLIB_AVAILABLE:
            # Parallel version using joblib with threading backend
            # Threading is recommended because NumPy/SciPy release the GIL
            # and we use shared memory (no serialization overhead)
            results = Parallel(n_jobs=cpu, backend='threading', verbose=0)(
                delayed(self._process_single_pixel)(i, p0, x, y, mask) 
                for i, p0 in enumerate(phi0s)
            )
            
            # Unpack results
            for i, (a_val, b_val) in enumerate(results):
                a[i] = a_val
                b[i] = b_val
        else:
            # Sequential version (original or fallback if joblib not available)
            progress_interval = max(1, npx // 20)  # Mostrar progreso cada 5%
            
            # Loop over all pixels
            # i is the same as theta_k in the PACO paper
            for i,p0 in enumerate(phi0s):
                # Mostrar progreso
                if i % progress_interval == 0:
                    progress = (i / npx) * 100
                    print(f"  Progreso: {progress:.1f}% ({i}/{npx} píxeles)", end='\r')
                
                a_val, b_val = self._process_single_pixel(i, p0, x, y, mask)
                a[i] = a_val
                b[i] = b_val
        
        print(f"\nDone (procesados {npx} píxeles)")
        return a,b
    
    def _process_single_pixel(self, i, p0, x, y, mask):
        """
        Process a single pixel position (helper function for parallelization)
        
        This function is extracted from the main loop to enable parallelization.
        Each pixel calculation is independent, making this perfectly parallelizable.
        
        Parameters
        ----------
        i : int
            Index of the pixel (for progress tracking, not used in parallel version)
        p0 : tuple
            Pixel coordinates to process
        x, y : np.ndarray
            Coordinate grids
        mask : np.ndarray
            Circular mask for patch extraction
            
        Returns
        -------
        a_val : float
            Value of a for this pixel
        b_val : float
            Value of b for this pixel
        """
        # Get list of pixels for each rotation angle
        angles_px = getRotatedPixels(x, y, p0, self.m_angles)
        
        # Create arrays for this pixel
        patch = np.zeros((self.m_nFrames, self.m_nFrames, self.m_psf_area))
        m = np.zeros((self.m_nFrames, self.m_psf_area))
        Cinv = np.zeros((self.m_nFrames, self.m_psf_area, self.m_psf_area))
        h = np.zeros((self.m_nFrames, self.m_psf_area))
        
        # Iterate over each temporal frame/each angle
        # Same as iterating over phi_l
        for l, ang in enumerate(angles_px):
            patch[l] = self.getPatch(ang, self.m_pwidth, mask)  # Get the column of patches at this point
            m[l], Cinv[l] = pixelCalc(patch[l])
            h[l] = self.m_psf[mask]

        # Calculate a and b, matrices
        a_val = self.al(h, Cinv)
        b_val = self.bl(h, Cinv, patch, m)
        
        return a_val, b_val
  
