#!/usr/bin/env python3
"""
REFPROP Interface Module
Provides thermodynamic property calculations using NIST REFPROP via ctREFPROP
"""

import numpy as np
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass

try:
    from ctREFPROP.ctREFPROP import REFPROPFunctionLibrary
    import os
    
    # Initialize REFPROP - adjust path as needed for your system
    # Common paths:
    # Windows: r'C:\Program Files (x86)\REFPROP'
    # Linux: '/usr/local/REFPROP'
    # Mac: '/Applications/REFPROP.app/Contents/Resources'
    
    REFPROP_PATH = os.environ.get('RPPREFIX', r'C:\Program Files (x86)\REFPROP')
    
    try:
        RP = REFPROPFunctionLibrary(REFPROP_PATH)
        RP.SETPATHdll(REFPROP_PATH)
        
        # Test if REFPROP is working by trying to load a simple fluid
        units = RP.GETENUMdll(0, "MASS BASE SI").iEnum
        REFPROP_AVAILABLE = True
        
    except Exception as e:
        print(f"Warning: REFPROP initialization failed: {e}")
        REFPROP_AVAILABLE = False
        RP = None
        
except ImportError:
    REFPROP_AVAILABLE = False
    print("Warning: ctREFPROP not available. Install with: pip install ctREFPROP")
    print("Also ensure NIST REFPROP is installed on your system")
    RP = None


@dataclass
class RefrigerantProperties:
    """Enhanced fluid properties from REFPROP"""
    name: str
    T: float  # Temperature [K]
    P: float  # Pressure [Pa]
    # Liquid properties
    rho_l: float  # Liquid density [kg/m³]
    mu_l: float   # Liquid viscosity [Pa·s]
    k_l: float    # Liquid thermal conductivity [W/m·K]
    cp_l: float   # Liquid specific heat [J/kg·K]
    Pr_l: float   # Liquid Prandtl number
    # Vapor properties
    rho_v: float  # Vapor density [kg/m³]
    mu_v: float   # Vapor viscosity [Pa·s]
    k_v: float    # Vapor thermal conductivity [W/m·K]
    cp_v: float   # Vapor specific heat [J/kg·K]
    # Phase change properties
    h_fg: float   # Latent heat of vaporization [J/kg]
    sigma: float  # Surface tension [N/m]
    T_sat: float  # Saturation temperature at given pressure [K]
    P_sat: float  # Saturation pressure at given temperature [Pa]
    # Critical properties
    T_crit: float # Critical temperature [K]
    P_crit: float # Critical pressure [Pa]
    # Quality
    quality: Optional[float] = None  # Vapor quality (0-1) if two-phase


class RefpropInterface:
    """Interface for REFPROP fluid property calculations"""
    
    # Common refrigerants and their REFPROP names
    REFRIGERANT_MAP = {
        'r134a': 'R134A',
        'r1234yf': 'R1234YF',
        'r1234ze': 'R1234ZE',
        'r245fa': 'R245FA',
        'r410a': 'R410A',
        'r32': 'R32',
        'r404a': 'R404A',
        'r507a': 'R507A',
        'r22': 'R22',
        'r744': 'CO2',  # CO2
        'r717': 'AMMONIA',  # Ammonia
        'r718': 'WATER',  # Water
        'water': 'WATER',
        'co2': 'CO2',
        'ammonia': 'AMMONIA',
        'propane': 'PROPANE',
        'r290': 'PROPANE',
        'isobutane': 'ISOBUTAN',
        'r600a': 'ISOBUTAN'
    }
    
    def __init__(self):
        """Initialize REFPROP interface"""
        if not REFPROP_AVAILABLE:
            raise RuntimeError("REFPROP is not available. Please install ctREFPROP and NIST REFPROP")
        
        self.RP = RP
        self._current_fluid = None
        self._fluid_loaded = False
    
    def set_fluid(self, fluid_name: str) -> bool:
        """
        Set the working fluid for calculations
        
        Args:
            fluid_name: Name of fluid (case-insensitive)
            
        Returns:
            True if successful, False otherwise
        """
        fluid_key = fluid_name.lower()
        
        if fluid_key not in self.REFRIGERANT_MAP:
            print(f"Warning: Unknown fluid '{fluid_name}'. Available fluids:")
            for key in sorted(self.REFRIGERANT_MAP.keys()):
                print(f"  - {key}")
            return False
        
        refprop_name = self.REFRIGERANT_MAP[fluid_key]
        
        # Set up fluid in REFPROP using SETFLUIDSdll (more reliable)
        ierr = self.RP.SETFLUIDSdll(refprop_name)
        
        if ierr.ierr != 0:
            print(f"Error loading fluid {refprop_name}: {ierr.herr}")
            return False
        
        self._current_fluid = fluid_key
        self._fluid_loaded = True
        return True
    
    def get_saturation_properties(self, T: Optional[float] = None, 
                                 P: Optional[float] = None) -> RefrigerantProperties:
        """
        Get saturation properties at given temperature or pressure
        
        Args:
            T: Temperature [K] (provide T or P, not both)
            P: Pressure [Pa] (provide T or P, not both)
            
        Returns:
            RefrigerantProperties object with saturation properties
        """
        if not self._fluid_loaded:
            raise RuntimeError("No fluid loaded. Call set_fluid() first")
        
        if T is not None and P is not None:
            raise ValueError("Provide either T or P, not both")
        
        if T is None and P is None:
            raise ValueError("Must provide either T or P")
        
        # Get saturation properties
        if T is not None:
            # Given temperature, find saturation pressure
            # Get liquid properties (Q=0)
            props_l = self.RP.TQFLSHdll(T, 0, [1.0])
            if props_l.ierr != 0:
                raise RuntimeError(f"REFPROP error: {props_l.herr}")
            
            # Get vapor properties (Q=1)
            props_v = self.RP.TQFLSHdll(T, 1, [1.0])
            if props_v.ierr != 0:
                raise RuntimeError(f"REFPROP error: {props_v.herr}")
            
            P_sat = props_l.P * 1000  # Convert kPa to Pa
            T_sat = T
            
        else:  # P is given
            # Given pressure, find saturation temperature
            P_kPa = P / 1000.0  # Convert Pa to kPa for REFPROP
            
            # Get liquid properties (Q=0)
            props_l = self.RP.PQFLSHdll(P_kPa, 0, [1.0])
            if props_l.ierr != 0:
                raise RuntimeError(f"REFPROP error: {props_l.herr}")
            
            # Get vapor properties (Q=1)
            props_v = self.RP.PQFLSHdll(P_kPa, 1, [1.0])
            if props_v.ierr != 0:
                raise RuntimeError(f"REFPROP error: {props_v.herr}")
            
            T_sat = props_l.T
            P_sat = P
        
        # Get transport properties for liquid
        trans_l = self.RP.TRNPRPdll(props_l.T, props_l.D, [1.0])
        if trans_l.ierr != 0:
            raise RuntimeError(f"REFPROP transport error: {trans_l.herr}")
        
        # Get transport properties for vapor
        trans_v = self.RP.TRNPRPdll(props_v.T, props_v.D, [1.0])
        if trans_v.ierr != 0:
            raise RuntimeError(f"REFPROP transport error: {trans_v.herr}")
        
        # Get surface tension
        sigma_result = self.RP.SURFTdll(T_sat, props_l.D, props_v.D, [1.0], [1.0])
        sigma = sigma_result.value / 1000.0  # Convert mN/m to N/m
        
        # Get critical properties
        info = self.RP.INFOdll(1)
        T_crit = info.Tc
        P_crit = info.Pc * 1000  # Convert kPa to Pa
        
        # Calculate latent heat
        h_l = props_l.h * 1000  # Convert kJ/kg to J/kg
        h_v = props_v.h * 1000  # Convert kJ/kg to J/kg
        h_fg = h_v - h_l
        
        # Calculate Prandtl numbers
        Pr_l = props_l.cp * trans_l.eta / trans_l.tcx
        Pr_v = props_v.cp * trans_v.eta / trans_v.tcx
        
        return RefrigerantProperties(
            name=self._current_fluid,
            T=T_sat,
            P=P_sat,
            # Liquid properties
            rho_l=props_l.D,
            mu_l=trans_l.eta / 1e6,  # Convert μPa·s to Pa·s
            k_l=trans_l.tcx / 1000,   # Convert mW/m·K to W/m·K
            cp_l=props_l.cp * 1000,  # Convert kJ/kg·K to J/kg·K
            Pr_l=Pr_l,
            # Vapor properties
            rho_v=props_v.D,
            mu_v=trans_v.eta / 1e6,   # Convert μPa·s to Pa·s
            k_v=trans_v.tcx / 1000,    # Convert mW/m·K to W/m·K
            cp_v=props_v.cp * 1000,  # Convert kJ/kg·K to J/kg·K
            # Phase change properties
            h_fg=h_fg,
            sigma=sigma,
            T_sat=T_sat,
            P_sat=P_sat,
            # Critical properties
            T_crit=T_crit,
            P_crit=P_crit
        )


# Convenience function for backward compatibility
def get_refrigerant_properties(fluid_name: str, T: Optional[float] = None, 
                              P: Optional[float] = None) -> RefrigerantProperties:
    """
    Quick access to refrigerant properties
    
    Args:
        fluid_name: Name of refrigerant
        T: Temperature [K] (for saturation properties)
        P: Pressure [Pa] (for saturation properties)
        
    Returns:
        RefrigerantProperties object
    """
    if not REFPROP_AVAILABLE:
        raise RuntimeError("REFPROP is not available")
    
    interface = RefpropInterface()
    interface.set_fluid(fluid_name)
    
    if T is not None or P is not None:
        return interface.get_saturation_properties(T=T, P=P)
    else:
        # Return properties at standard conditions (25°C, 1 atm)
        return interface.get_saturation_properties(P=101325)


if __name__ == "__main__":
    # Test the REFPROP interface
    if REFPROP_AVAILABLE:
        print("REFPROP is available!")
        
        # Test with R134a
        interface = RefpropInterface()
        interface.set_fluid('r134a')
        
        # Get saturation properties at 10°C
        T = 283.15  # 10°C in K
        props = interface.get_saturation_properties(T=T)
        
        print(f"\nR134a saturation properties at {T-273.15:.1f}°C:")
        print(f"  Saturation pressure: {props.P_sat/1000:.2f} kPa")
        print(f"  Liquid density: {props.rho_l:.2f} kg/m³")
        print(f"  Vapor density: {props.rho_v:.2f} kg/m³")
        print(f"  Latent heat: {props.h_fg/1000:.1f} kJ/kg")
        print(f"  Surface tension: {props.sigma*1000:.2f} mN/m")
        
        # Test with water
        interface.set_fluid('water')
        props_water = interface.get_saturation_properties(T=373.15)  # 100°C
        print(f"\nWater saturation properties at 100°C:")
        print(f"  Saturation pressure: {props_water.P_sat/1000:.2f} kPa")
        print(f"  Latent heat: {props_water.h_fg/1000:.1f} kJ/kg")
    else:
        print("REFPROP is not available. Please install ctREFPROP and NIST REFPROP")