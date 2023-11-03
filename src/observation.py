'''The :mod:`observation` module contains the observational data used in the analysis.
The module contains a :mod:`numpy` structured array, :data:`observation.raw`, which is
intended to be accessed through the :class:`Dwarf` class. The array contains the 
following information for each dwarf:

 - The distance modulus [1]_,  
 - The half-light radius (in arcminutes) [1]_,  
 - The ellipticity [1]_,  
 - The heliocentric line-of-sight velocity dispersion (in km/s) [1]_,  
 - The base-ten logarithm of the V-band luminosity (in :math:`L_\odot`) [2]_,  
 - The pericentric distance (in kpc) [3]_,  
 - The infall time (in Gyr lookback) [4]_, and  
 - The present-day galactocentric distance (in kpc) [2]_.  

Each of these (save the galactocentric distance) has an associated upper and lower uncertainty, 
corresponding to the 16th and 85th percentiles for the value. 

References
----------
.. [1] Battaglia+22, `2022A&A...657A..54B <https://ui.adsabs.harvard.edu/abs/2022A%26A...657A..54B/abstract>`_
.. [2] Muñoz+18, `2018ApJ...860...66M <https://ui.adsabs.harvard.edu/abs/2018ApJ...860...66M/abstract>`_
.. [3] Pace+22, `2022ApJ...940..136P <https://ui.adsabs.harvard.edu/abs/2022ApJ...940..136P/abstract>`_
.. [4] Fillingham+19, `2019arXiv190604180F <https://ui.adsabs.harvard.edu/abs/2019arXiv190604180F/abstract>`_

'''

import numpy as np
from astropy import constants as const
from astropy import units as u
from scipy import stats

raw = np.array([
    #                            [  distance mod  ] [     r-half     ] [     ellip     ] [  sigma LOS  ] [     logLv     ] [     rperi     ] [   tinfall   ] [RGC]
    ('Canes Venatici I', 'CVnI', 21.62, 0.05, 0.05,  7.48, 0.2 , 0.2 , 0.45, 0.02, 0.02,  7.6, 0.4, 0.4, 5.45, 0.02, 0.02, 84.5, 53.6, 37.2,  9.4, 0.9, 2.3, 217.8),
    ('Carina', 'Car',            20.12, 0.12, 0.12, 10.2 , 0.1 , 0.1 , 0.37, 0.01, 0.01,  6.6, 1.2, 1.2, 5.7 , 0.02, 0.02, 77.9, 24.1, 17.9,  9.9, 0.6, 2.7, 106.7),
    ('Draco', 'Dra',             19.53, 0.07, 0.07,  9.61, 0.1 , 0.1 , 0.3 , 0.01, 0.01,  9. , 0.3, 0.3, 5.42, 0.02, 0.02, 58. , 11.4,  9.5, 10.4, 2.4, 3.1,  76. ),
    ('Fornax', 'Fnx',            20.72, 0.04, 0.04, 18.5 , 0.1 , 0.1 , 0.3 , 0.01, 0.01, 11.4, 0.4, 0.4, 7.32, 0.06, 0.06, 76.7, 43.1, 27.9, 10.7, 0.8, 3.1, 149.1),
    ('Leo I', 'LeoI',            22.15, 0.1 , 0.1 ,  3.53, 0.03, 0.03, 0.31, 0.01, 0.01,  9.2, 0.4, 0.4, 6.64, 0.11, 0.11, 47.5, 30.9, 24. ,  2.3, 0.6, 0.5, 256. ),
    ('Leo II', 'LeoII',          21.68, 0.11, 0.11,  2.46, 0.03, 0.03, 0.07, 0.02, 0.02,  7.4, 0.4, 0.4, 5.83, 0.02, 0.02, 61.4, 62.3, 34.7,  7.8, 3.3, 2. , 235.7),
    ('Sextans', 'Sxt',           19.64, 0.01, 0.01, 21.4 , 0.7 , 0.6 , 0.27, 0.03, 0.03,  8.4, 0.4, 0.4, 5.51, 0.04, 0.04, 82.2,  3.8,  4.3,  8.4, 2.7, 0.9,  89.2),
    ('Sculptor', 'Scl',          19.62, 0.04, 0.04, 12.43, 0.18, 0.18, 0.36, 0.01, 0.01, 10.1, 0.3, 0.3, 6.26, 0.06, 0.06, 44.9,  4.3,  3.9,  9.9, 1.7, 2.9,  86.1),
    ('Ursa Minor', 'UMi',        19.41, 0.12, 0.12, 18.2 , 0.1 , 0.1 , 0.55, 0.01, 0.01,  8. , 0.3, 0.3, 5.54, 0.02, 0.02, 55.7,  8.4,  7. , 10.7, 1.7, 2. ,  78. )],
    dtype=[('name', '<U16'), ('abbr', '<U6'),                           # Dwarf name and abbreviation
    ('distmod', '<f8'), ('distmod_hi', '<f8'), ('distmod_lo', '<f8'),   # distance modulus                        [Battaglia+22  Table B.1]
    ('rh', '<f8'), ('rh_hi', '<f8'), ('rh_lo', '<f8'),                  # half-light radius (arcmin)              [Battaglia+22  Table B.1]
    ('ellip', '<f8'), ('ellip_hi', '<f8'), ('ellip_lo', '<f8'),         # ellipticity                             [Battaglia+22  Table B.1]
    ('sigvlos', '<f8'), ('sigvlos_hi', '<f8'), ('sigvlos_lo', '<f8'),   # heliocentric l.o.s. velocity dispersion [Battaglia+22  Table B.1]
    ('logLv', '<f8'), ('logLv_hi', '<f8'), ('logLv_lo', '<f8'),         # V-band luminosity                       [Muñoz+18      Table 6]
    ('rperi', '<f8'), ('rperi_hi', '<f8'), ('rperi_lo', '<f8'),         # pericentric distance                    [Pace+22       Table 3]
    ('tinfall', '<f8'), ('tinfall_hi', '<f8'), ('tinfall_lo', '<f8'),   # infall time                             [Fillingham+19 Table 1]
    ('RGC', '<f8')])                                                    # galactocentric radius                   [Muñoz+18      Table 1]

_namelist = [
    ['canesvenatici', 'canesvenaticii', 'canesvenatici1', 'cvn', 'cvni', 'cvn1'],
    ['carina', 'car'],
    ['draco', 'dra'],
    ['fornax', 'fnx'],
    ['leoi', 'leo'],
    ['leoii'],
    ['sextans', 'sxt', 'sex'],
    ['sculptor', 'scl'],
    ['ursaminor', 'umi'],
]

_namemap = {name : i for i, lst in enumerate(_namelist) for name in lst}
dwarf_names = raw['name']

def idx(dwarf_name):
    '''Finds the index corresponding to a dwarf in the :data:`observation.raw` array, 
    allowing for capitalization, spaces, and common abbreviations.

    Parameters
    ----------
    dwarf_name : str
                 The name of the dwarf.

    Returns
    -------
    int
        The index into :data:`observation.raw` corresponding to the dwarf.

    Raises
    ------
    KeyError
        If ``dwarf_name`` is unrecognized. 
    '''
    norm_name = dwarf_name.replace(" ", "").lower()
    try:
        return _namemap[norm_name]
    except KeyError as e:
        try: # python 3.11 way 
            e.add_note(f'"{dwarf_name}" is not a recognized dwarf or abbreviation.')
            raise
        except AttributeError: # for older versions of python
            raise KeyError(f'"{dwarf_name}" is not a recognized dwarf or abbreviation.') from e


class Dwarf:
    r'''Interprets the raw observational data to allow easy access to probability 
    distributions of known parameters.

    PDFs are constructed internally as two-sided Gaussians to account for asymmetric
    observational uncertainties. For an observational value :math:`\mu_{-\sigma_{\downarrow}}^{+\sigma_{\uparrow}}`,
    the PDF is

    .. math:: f_i(\theta) = \frac{2}{\sigma_\uparrow + \sigma_\downarrow} \times \begin{cases}\operatorname{Norm}(\theta;\;\mu,\,\sigma_\downarrow) & \theta < \mu \\ \operatorname{Norm}(\theta;\;\mu,\,\sigma_\uparrow) & \theta \geq \mu \end{cases}

    with 

    .. math:: \operatorname{Norm}(x;\;\mu,\,\sigma) = \frac{1}{\sqrt{2\pi}\sigma}\exp\left[-\frac{1}{2}\left(\frac{x - \mu}{\sigma}\right)^2\right]

    Attributes
    ----------
        name : str
            The full name of the dwarf
        rhalf : float
            The 3D sphericalized, deprojected half-light radius in pc, computed as per [1]_.
        logMhalf : float
            The base-ten logarithm of the mass in :math:`M_\odot` within :attr:`rhalf`, as determined by the Wolf+16 estimator [2]_.
        logMstar : float
            The base-ten logarithm of the stellar mass in :math:`M_\odot`, computed via a mass-to-light ratio.
    
    Methods
    -------
    logMstar_pdf(logMstar)
        Evaluates the observed PDF for :math:`\log_{10}M_\star` at the value ``logMstar``
    logMhalf_pdf(logMhalf)
        Evaluates the observed PDF for :math:`\log_{10}M_{1/2}` at the value ``logMhalf``


    Notes
    -----
    For each physical quantity (:attr:`rhalf`, :attr:`logMhalf`, etc.), there is also an associated ``<quantity>_err`` 
    attribute containing the lower and upper error bars, both as non-negative floats. These are used to compute the two-sided Gaussian PDFs.
    
    References
    ----------
    .. [1] Sanders-Evans16 `2016ApJ...830L..26S <https://ui.adsabs.harvard.edu/abs/2016ApJ...830L..26S>`_
    .. [2] Wolf+16 `2010MNRAS.406.1220W <https://ui.adsabs.harvard.edu/abs/2010MNRAS.406.1220W>`_
    '''
    def __init__(self, name, ML_ratio = 1.2, ML_error = 0, logMstar_scatter = 0.16):
        '''Constructs the :class:`Dwarf` object from the :data:`observation.raw` data. 

        In particular, it converts the angular on-sky information into a physical size :math:`r_{1/2}`, which is used in the Wolf+16 estimator 
        (along with the velocity dispersion) to determine the :math:`M_{1/2}` of the dwarf. Further, it converts the V-band luminosity into
        a stellar mass through the provided mass-to-light ratio. Uncertainties are propagated throughout. 

        Parameters
        ----------
        name : str
            The name of the dwarf
        ML_ratio : float, optional
            The mass-to-light ratio to assume for this dwarf. The default is 1.2.
        ML_error : float, optional
            An uncertainty on the mass-to-light ratio, propagated through to the stellar mass. The default is zero.
        logMstar_scatter : float, optional
            An additional uncertainty (in dex) to add to the stellar mass. Similar to :attr:`ML_error`, but added directly instead of being propagated. The default is 0.16.
        '''
        myID = idx(name)
        data = raw[myID]
        self.name = data['name']

        # convert angular information into 2D half-light radius Re
        distance = 10**(data['distmod']/5 + 1) * u.pc
        distmod_err = np.array([data['distmod_lo'], data['distmod_hi']])
        distance_err = distance * distmod_err * np.log(10)/5

        major_axis = (distance * data['rh'] * u.arcmin).to(u.pc, u.dimensionless_angles())
        angular_size_err = np.array([data['rh_lo'], data['rh_hi']])
        major_axis_err = np.hypot(distance_err/distance, angular_size_err/data['rh']) * major_axis

        Re = major_axis * np.sqrt(1 - data['ellip']) # Sphericalize as per Sanders and Evans
        ellipticity_error = np.array([data['ellip_lo'], data['ellip_hi']])
        Re_err = Re * np.hypot(major_axis_err/major_axis, ellipticity_error/(2 - 2 * data['ellip']))

        # find Mhalf using the Wolf et al. estimator
        dispersion = data['sigvlos'] * u.km/u.s
        dispersion_err = np.array([data['sigvlos_lo'], data['sigvlos_hi']]) * u.km/u.s
        Mhalf = (4 * dispersion**2 * Re / const.G).to(u.Msun).value
        Mhalf_err = Mhalf * np.hypot(2 * dispersion_err/dispersion, Re_err/Re).to(u.dimensionless_unscaled).value

        self.logMhalf = np.log10(Mhalf)
        self.logMhalf_err = Mhalf_err/(Mhalf * np.log(10))
        
        # convert to 3D here 
        self.rhalf = Re.to(u.pc) * 4/3
        self.rhalf_err = Re_err.to(u.pc) * 4/3

        # map Lv -> Mstar using mass to light ratio (with error) plus additional logMstar error (in dex)
        self.logMstar = data['logLv'] + np.log10(ML_ratio)
        #                               measurement uncertainty in Lv       uncertainty in mass-to-light ratio   added scatter
        self.logMstar_err = np.array([data['logLv_lo'], data['logLv_hi']]) + ML_error/(ML_ratio * np.log(10)) + logMstar_scatter                   

        # last couple parameters
        self.rperi = data['rperi']
        self.rperi_err = np.array([data['rperi_lo'], data['rperi_hi']])
        self.tinfall = data['tinfall']
        self.tinfall_err = np.array([data['tinfall_lo'], data['tinfall_hi']])
    
    @staticmethod
    def _twosided(x, mean, lowerr, higherr):
        scale = 2/(lowerr + higherr)
        lowgauss = scale * stats.norm.pdf((x-mean)/lowerr)
        highgauss = scale * stats.norm.pdf((x-mean)/higherr)
        return np.where(x < mean, lowgauss, highgauss)
  
    def logMstar_pdf(self, logMstar):
        '''Evaluates the PDF for :math:`f_i(\log_{10}M_\star)` at the value :attr:`logMstar`

        Parameters
        ----------
        logMstar : float
            the value at which to evaluate the PDF.
        '''
        return self._twosided(logMstar, self.logMstar, *self.logMstar_err)

    def logMhalf_pdf(self, logMhalf):
        '''Evaluates the PDF for :math:`f_i(\log_{10}M_{1/2})` at the value :attr:`logMhalf`

        Parameters
        ----------
        logMhalf : float
            the value at which to evaluate the PDF.
        '''
        return self._twosided(logMhalf, self.logMhalf, *self.logMhalf_err)

    def __repr__(self):
        return f'{self.name}: logMstar = {self.logMstar:0.2f} ± {self.logMstar_err.mean():0.2f} dex, logMhalf = {self.logMhalf:0.2f} ± {self.logMhalf_err.mean():0.2f} dex @ r1/2 = {self.rhalf.to(u.pc).value:0.0f} pc'
