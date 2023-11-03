r'''The :mod:`analysis` module contains the components of the inference related to ``SatGen``:

1. The :class:`SatGenData` class acts as a wrapper around the processed ``SatGen`` output, and holds the direct sampling of :math:`f_\mathrm{pred}(\boldsymbol{x},\,\boldsymbol{\theta})` (i.e., the full parameter information of each ``SatGen`` satellite) as well as the interpolated :math:`f_\mathrm{pred}(\boldsymbol{\theta})` distributions.
2. The :class:`Weights` class combines these with observed values to compute the weight applied to each ``SatGen`` satellite for a given choice of :math:`\boldsymbol{\theta}`.

'''

import numpy as np
from scipy import stats
from scipy import interpolate as interp
from scipy import optimize as opt
from SatGen.profiles import Dekel
from astropy import units as u

import src.observation as obs

class SatGenData:
    r'''Acts as a wrapper around the processed ``SatGen`` output.

    This class holds both the raw ``SatGen`` data, which samples the distribution :math:`f_\mathrm{pred}(\boldsymbol{x},\,\boldsymbol{\theta})`, 
    and the interpolated :math:`f_\mathrm{pred}(\boldsymbol{\theta})` distributions. Generally, you will only need to access attributes corresponding
    to the parameters you want to infer (e.g., :math:`r_\mathrm{max}`, :math:`v_\mathrm{max}`), and you will use a :class:`Weights` object to interface with
    the internals of the class.

    Attributes
    ----------
    profiles : list(SatGen.profiles.Dekel)
        The profiles of the ``SatGen`` satellites.
    logMstar : np.ndarray(float)
        Base-ten logarithm of the stellar mass (in :math:`M_\odot`) for the ``SatGen`` satellites.
    logMhalf : dict
        Maps from a dwarf name to an ``np.ndarray`` of :math:`M(r_{1/2})` (in :math:`M_\odot`) for the ``SatGen`` satellites, where :math:`r_{1/2}` is the observed value for the MW dwarf.
    data : np.ndarray
        Contains full information about all the satellites in the ``SatGen`` file. All other attributes are ultimately derived from this array, but it is preferred to use :meth:`get` to access the data.
    hosts : np.ndarray
        Contains full information about the MW hosts of the satellites. :attr:`data` has a ``hostID`` key which maps into this array.
    surviving, calibrated, splashback  : np.ndarray(bool)
        Used to select satellites from the :attr:`data` array which are relevant to the study, i.e., those that are above the mass resolution, within the ``SatGen`` calibrated regime (with regard to mass loss and innermost slope), and within the MW's virial radius, respectively. 
    mask : np.ndarray(bool)
        Used to select satellites from the :attr:`data` array which pass all the cuts implemented by :attr:`surviving`, :attr:`calibrated`, and :attr:`splashback` 

    Methods
    -------
    get(attr)
        Retrives the attribute ``attr`` from the :attr:`data` array, applying the :attr:`mask` cut to select only satellites relevant to the study.
    get_pdf(attr)
        Similar to :meth:`get`, but returns an interpolated histogram PDF for ``attr`` rather than the ``attr`` values themselves.
    logMstar_pdf(logMstar)
        Evaluates the PDF :math:`f_\mathrm{pred}(\log_{10}M_\star)` at ``logMstar``. Used in :class:`Weights`.
    logMhalf_pdf(dwarf_name, logMhalf)
        Evaluates the PDF :math:`f_\mathrm{pred}(\log_{10}M_{1/2})` at ``logMhalf``. Since the :math:`M_{1/2}` values depend on the radius :math:`r_{1/2}` at which it is measured, this PDF depends on the MW dwarf and its :math:`r_{1/2}` value. 
    joint_mass_pdf(dwarf_name, logMstar, logMhalf)
        Evaluates the joint PDF :math:`f_\mathrm{pred}(\log_{10}M_\star, \log_{10}M_{1/2})` at (``logMstar``, ``logMhalf``). Used in :class:`Weights`.

    '''
    def __init__(self, fname):
        '''Construct the :class:`SatGenData` object.
        
        Parameters
        ----------
        fname : str or list of str
            The name of the file to load, or a list of such names.
        
        '''
        # load in the SatGen data
        if type(fname) is str:
            with np.load(fname) as f:
                self.data = f['sats']
                self.hosts = f['hosts']
        elif type(fname) is list:
            datalist = []
            hostlist = []
            for name in fname:
                with np.load(name) as f:
                    data = f['sats']
                    hosts = f['hosts']
                datalist.append(data)
                hostlist.append(hosts)
            self.data = np.concatenate(datalist)
            self.hosts = np.concatenate(hostlist)
        else:
            raise TypeError('fname unknown')

        # cuts on the dataset: above the mass resolution, within tidal track calibration, physical density profile
        self.surviving = self.data['virial_mass'] > 10**6.5
        self.calibrated = (self.data['mass_loss'] > 0.01) & (self.data['DekelCAD'][:, 1] < 3)
        self.splashback = np.linalg.norm(self.data['position'], axis = 1) > self.hosts['virial_radius'][self.data['hostID']]
        self.mask = self.surviving & self.calibrated & ~self.splashback
        
        self.PDF_N_BINS = np.count_nonzero(self.mask)//20_000

        # save a few handy slices of the SatGen data to self attributes
        self.profiles = [Dekel(m, c, a, D) for (m, (c, a, D)) 
                            in zip(self.get('virial_mass'), self.get('DekelCAD'))]
        
        self.logMstar = self.get('stellar_mass', log = True)
        dwarf_rhalfs_kpc = [(obs.Dwarf(name).rhalf.to(u.kpc).value, name) for name in obs.dwarf_names]
        self.logMhalf = {name: np.log10([p.M(r) for p in self.profiles]) for r, name in dwarf_rhalfs_kpc}

        # set up the marginal PDFs for the SatGen data
        logMstar_pdf = _hist1D(self.logMstar, bins = self.PDF_N_BINS)
        logMhalf_pdf = {name: _hist1D(self.logMhalf[name], bins = self.PDF_N_BINS) for name in obs.dwarf_names}
        # todo: would like to make this KDE -- e.g. KDEpy.FFTKDE is very fast
        joint_masses = {name: _hist2D(self.logMstar, self.logMhalf[name], bins = self.PDF_N_BINS//2) for name in obs.dwarf_names}

        self._dist = dict(logMstar = logMstar_pdf, logMhalf = logMhalf_pdf, masses = joint_masses)

    def get(self, attr, log = False):
        '''Retrieve fields from the ``SatGen`` output data.
        
        Parameters
        ----------
        attr : str
            The name of the attribute to retrieve. This is taken from :attr:`data` with the :attr:`mask` applied to select only satellites relevant to the study.
        log : bool, optional
            If ``True``, return the base-ten logarithm of the ``attr`` values.
        
        Returns
        -------
        np.ndarray
            Values for the attribute for each ``SatGen`` satellite.
            
        '''
        values = self.data[attr][self.mask]
        return np.log10(values) if log else values
    
    def get_pdf(self, attr, log = False, bins = None):
        '''Retrieve a probability distribution describing a field from the ``SatGen`` output data.
        
        Parameters
        ----------
        attr : str
            The name of the attribute to retrieve a PDF for. This is taken from :attr:`data` with the :attr:`mask` applied to select only satellites relevant to the study.
        log : bool, optional
            If ``True``, return the PDF of the base-ten logarithm of the ``attr`` values.
        bins : optional
            Options passed to ``np.histogram``. Defaults to :attr:`PDF_N_BINS`, which is the number of ``SatGen`` satellites in the distribution divided by 20,000 (a value arbitrarily chosen to give good PDF resolution)
        
        Returns
        -------
        scipy.interpolate.RegularGridInterpolator
            Interpolator for the PDF.
            
        '''
        values = self.get(attr, log)
        if bins is None:
            bins = self.PDF_N_BINS
        return _hist1D(values, bins = bins)
            
    def logMstar_pdf(self, logMstar):
        '''Evaluates the PDF :math:`f_\mathrm{pred}(\log_{10}M_\star)` at ``logMstar``. Used in :class:`Weights`.
        
        Parameters
        ----------
        logMstar : float or list of float
            The value at which to evaluate the PDF
        
        Returns
        -------
        float or list of float
            Value of the PDF at ``logMstar``
            
        '''
        return self._dist['logMstar'](logMstar)

    def logMhalf_pdf(self, dwarf_name, logMhalf):
        '''Evaluates the PDF :math:`f_\mathrm{pred}(\log_{10}M(r_{1/2}))` at ``logMhalf``. Used in :class:`Weights`.
        
        Parameters
        ----------
        dwarf_name : str
            The name of the dwarf to use for :math:`r_{1/2}`.
        logMhalf : float or list of float
            The value at which to evaluate the PDF
        
        Returns
        -------
        float or list of float
            Value of the PDF at ``logMstar``
        
        Notes
        -----
        This PDF is that of the enclosed mass profile of the ``SatGen`` satellites evaluated at a MW dwarf's half-light radius :math:`r_{1/2}`. As such, it depends on the dwarf of interest.
            
        '''
        return self._dist['logMhalf'][dwarf_name](logMhalf)

    def joint_mass_pdf(self, dwarf_name, logMstar, logMhalf):
        '''Evaluates the PDF :math:`f_\mathrm{pred}(\log_{10}M_\star, \log_{10}M_{1/2})` at (``logMstar``, ``logMhalf``). Used in :class:`Weights`.
        
        Parameters
        ----------
        dwarf_name : str
            The name of the dwarf to use for :math:`r_{1/2}`.
        logMstar : float or list of float
            The value of :math:`\log_{10}M_\star` at which to evaluate the PDF
        logMshalf : float or list of float
            The value of :math:`\log_{10}M_{1/2}` at which to evaluate the PDF
        
        Returns
        -------
        float or list of float
            Value of the PDF at (``logMstar``, ``logMhalf``)
            
        '''
        return self._dist['masses'][dwarf_name](np.c_[logMstar, logMhalf])


class Weights:
    r'''Provides weights for ``SatGen`` output such that their distribution can approximate observed probability distributions.

    In detail, this class uses a :class:`SatGenData` object with sampled :math:`f_\mathrm{pred}(\boldsymbol{x},\,\boldsymbol{\theta})` distributions, as well
    as observed :math:`f_i(\boldsymbol{\theta})` distributions and provides easy access to :math:`f_i(\boldsymbol{\theta})/f_\mathrm{pred}(\boldsymbol{\theta})`,
    which acts as a weight on the ``SatGen`` output such that the PDF :math:`f_\mathrm{pred}(\boldsymbol{x},\,\boldsymbol{\theta})\cdot f_i(\boldsymbol{\theta})/f_\mathrm{pred}(\boldsymbol{\theta})`
    has a marginal distribution of :math:`\boldsymbol{\theta}` matching observation. The marginal distribution of :math:`\boldsymbol{x}` is the distribution 
    inferred from ``SatGen`` for the parameter :math:`\boldsymbol{x}`.

    Attributes
    ----------
    sats : SatGenData
        The ``SatGen`` satellites to use for :math:`f_\mathrm{pred}` distributions.
    dwarf : Dwarf
        The MW dwarf to use for :math:`f_i` distributions 
    Mstar, Mhalf, joint_mass : np.ndarray(float)
        The weights derived from selecting on :math:`M_\star`, :math:`M_{1/2}`, or both combined, respectively.

    Methods
    -------
    pdf(values)
        Returns an interpolated PDF for the inferred distribution of ``values``.
    cdf(values)
        Returns an interpolated CDF for the inferred distribution of ``values``.
    quantile(values)
        Returns an interpolated quantile function (inverse CDF) for the inferred distribution of ``values``.

    '''    
    def __init__(self, satgen_sim, dwarf):
        '''Construct the :class:`Weights` object.
        
        Parameters
        ----------
        sats : SatGenData
            The ``SatGen`` satellites to use for :math:`f_\mathrm{pred}` distributions.
        dwarf : str or Dwarf
            The MW dwarf (or name thereof) to use for :math:`f_i` distributions 
        
        '''
        self.sats = satgen_sim
        if isinstance(dwarf, str):
            dwarf = obs.Dwarf(dwarf)
        self.dwarf = dwarf
        name = dwarf.name
        logMhalf = self.sats.logMhalf[name]
        logMstar = self.sats.logMstar

        self.Mstar = self.dwarf.logMstar_pdf(logMstar)/self.sats.logMstar_pdf(logMstar)
        self.Mhalf = self.dwarf.logMhalf_pdf(logMhalf)/self.sats.logMhalf_pdf(name, logMhalf)
        obs_weight = self.dwarf.logMhalf_pdf(logMhalf) * self.dwarf.logMstar_pdf(logMstar)
        normalization = self.sats.joint_mass_pdf(name, logMstar, logMhalf)
        self.joint_mass = obs_weight/normalization
        self.weights = dict(Mstar = self.Mstar, Mhalf = self.Mhalf, joint_mass = self.joint_mass)
        
    def pdf(self, values, weight_name = 'joint_mass', bins = 500):
        '''Returns an interpolated PDF for the inferred distribution of ``values`` based on one of the three weighting methods.
        
        Parameters
        ----------
        values : str or list(float)
            A parameter name used in :meth:`SatGenData.get` or a list of parameter values for ``SatGen`` satellites.
        weight_nme : str, optional
            The kind of weight to use. Accepted values are `'Mstar'`, `'Mhalf'`, or `'joint_mass'`. Defaults to `'joint_mass'`.
        bins : optional
            Number of bins to interpolate between, passed to ``np.histogram``. Defaults to 500.
        
        Returns
        -------
        scipy.interpolate.RegularGridInterpolator
            Linearly interpolates the histogram for the 1D PDF of ``values`` with the weights described by ``weight_name``.
            
        '''
        if values is str:
            values = self.sats.get(values)
        return _hist1D(values, self.weights[weight_name], bins)
    
    def cdf(self, values, weight_name = 'joint_mass'):
        '''Returns an interpolated CDF for the inferred distribution of ``values`` based on one of the three weighting methods.
        
        Parameters
        ----------
        values : str or list(float)
            A parameter name used in :meth:`SatGenData.get` or a list of parameter values for ``SatGen`` satellites.
        weight_nme : str, optional
            The kind of weight to use. Accepted values are `'Mstar'`, `'Mhalf'`, or `'joint_mass'`. Defaults to `'joint_mass'`.
        
        Returns
        -------
        scipy.interpolate.RegularGridInterpolator
            Linearly interpolates the CDF of ``values`` with the weights described by ``weight_name``.
            
        '''
        this_pdf = self.pdf(values, weight_name)
        xax = np.linspace(this_pdf.grid[0].min(), this_pdf.grid[0].max(), len(this_pdf.grid[0]) * 2)
        yax = np.cumsum(this_pdf(xax))
        yax /= yax[-1]
        cdf = interp.RegularGridInterpolator(xax.reshape((1,)+xax.shape), yax, bounds_error = False, fill_value = None, method = 'linear')
        return cdf
    
    def quantile(self, values, weight_name = 'joint_mass'):
        '''Returns an interpolated quantile function (inverse CDF) for the inferred distribution of ``values`` based on one of the three weighting methods.
        
        Parameters
        ----------
        values : str or list(float)
            A parameter name used in :meth:`SatGenData.get` or a list of parameter values for ``SatGen`` satellites.
        weight_nme : str, optional
            The kind of weight to use. Accepted values are `'Mstar'`, `'Mhalf'`, or `'joint_mass'`. Defaults to `'joint_mass'`.
        
        Returns
        -------
        scipy.interpolate.InterpolatedUnivariateSpline
            Linearly interpolates the quantile function of ``values`` with the weights described by ``weight_name``.
            
        '''
        this_pdf = self.pdf(values, weight_name)
        xax = np.linspace(this_pdf.grid[0].min(), this_pdf.grid[0].max(), len(this_pdf.grid[0]) * 2)
        pdf_vals = this_pdf(xax)
        yax = np.cumsum(pdf_vals)
        yax /= yax[-1]
        # want to map from cdf value -> grid value, but large weights mean large steps in the cdf
        # which fail the vertical line test. to avoid this, skip grid values that give the same cdf
        mask = np.concatenate([[True], np.diff(yax) > 1e-10])
        quantile = interp.InterpolatedUnivariateSpline(yax[mask], xax[mask], k = 1, ext = 'const', check_finite = True)
        return quantile
        

def _hist1D(valuesx, weights = None, bins = 150):
    '''Create a 1D PDF for ``valuesx`` by binning into a histogram and linearly interpolating between bins.
    
    Parameters
    ----------
    valuesx : list 
        The set of values to generate a PDF for.
    weights : list, optional
        The weights to apply to valuesx. The arrays must have the same length. Defaults to None, indicating equal weight.
    bins : int, optional
        The number of bins to use in the histogram, used as input to ``np.histogram``. Defaults to 150.
        
    Returns
    -------
    scipy.interpolate.RegularGridInterpolator
        Linearly interpolates the histogram for the 1D PDF of ``valuesx``.
    
    '''
    hist, bins = np.histogram(valuesx, bins = bins, density = True, weights = weights)
    # create array for center-of-bin values
    binwidth = np.diff(bins)[0]/2
    xax = np.concatenate(([bins.min() - binwidth], bins + binwidth))
    # pad around the outside of the given values to give zero in regions of no support
    yax = np.concatenate(([0], hist, [0]))
    # 1D linear interpolation preserves area, so it's okay to use it here
    # pdf = interp.interp1d(xax, yax, bounds_error = False, fill_value = 0, kind = 'linear')
    pdf = interp.RegularGridInterpolator(xax.reshape((1,)+xax.shape), yax, bounds_error = False, fill_value = 0, method = 'linear')
    return pdf

def _hist2D(valuesx, valuesy, weights = None, bins = (150, 150)):
    '''Create a 2D PDF for the joint distribution of ``valuesx`` and ``valuesy`` by binning into a histogram.
    
    Parameters
    ----------
    valuesx, valuesy : list 
        The x- and y values for points from which the PDF is generated.
    weights : list, optional
        The weights to apply to the points. This must have the same length as ``valuesx`` and ``valuesy``. Defaults to None, indicating equal weight.
    bins : int, optional
        The number of bins to use in the histogram, used as input to ``np.histogram2d``. Defaults to 150 in each direction.
    
    Returns
    -------
    scipy.interpolate.RegularGridInterpolator
        Interpolates the 2D histogram for the joint PDF of (``valuesx``, ``valuesy``). This uses the ``'nearest'`` method to preserve PDF volume.
    
    '''
    xlim = (valuesx.min(), valuesx.max())
    ylim = (valuesy.min(), valuesy.max())
    H, x, y = np.histogram2d(valuesx, valuesy, range = (xlim, ylim), bins = bins, density = True, weights = weights)
    # pad histogram edges with zeros
    interpvals = np.zeros(((len(x) + 1), (len(y) + 1)))
    interpvals[1:-1, 1:-1] = H
    # adjust the bin edges x and y to sit in the center of the bin
    binwidthx, binwidthy = np.diff(x)[0]/2, np.diff(y)[0]/2
    interpx = np.concatenate([[xlim[0] - 1], x[:-1] + binwidthx, [xlim[1] + 1]])
    interpy = np.concatenate([[ylim[0] - 1], y[:-1] + binwidthy, [ylim[1] + 1]])
    # must use "nearest" to preserve integral 
    # TODO: either promote the histogram to a KDE or find another way to smoothly interpolate the histogram while preserving area
    pdf = interp.RegularGridInterpolator((interpx, interpy), interpvals, 
                                         bounds_error = False, fill_value = 0, method = 'nearest')
    return pdf
