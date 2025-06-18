#!/usr/bin/env python3
"""
GPU Junction to Surface Thermal Calculator
Core thermal analysis calculations using modular data system

This script calculates temperature distribution from GPU junction to surface
using thermal resistance network modeling, accounting for:
- Material properties and temperature dependencies
- Heat transfer mechanisms (conduction, convection, phase change)
- Power dissipation models (dynamic and leakage power)
- Various cooling methods (air, liquid, evaporative, vapor chambers)
"""

import sys
import os
# Add the parent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass
from enum import Enum

# Import our data puller module
from main.utility.data_puller import DataPuller, MaterialProperties, LayerInfo, FluidProperties

# Import REFPROP interface for refrigerant properties
try:
    from main.utility.refprop_interface import RefpropInterface, RefrigerantProperties, REFPROP_AVAILABLE
except (ImportError, RuntimeError) as e:
    print(f"Warning: REFPROP interface not available: {e}")
    print("Refrigerant calculations will be limited to fluids in the JSON database.")
    REFPROP_AVAILABLE = False
    RefpropInterface = None
    RefrigerantProperties = None


class CoolingType(Enum):
    """
    Enumeration of cooling methods available for thermal analysis
    
    AIR: Natural or forced air convection cooling
    LIQUID: Single-phase liquid cooling (water, coolant)
    EVAPORATIVE: Two-phase cooling using boiling/evaporation
    VAPOR_CHAMBER: Heat pipes and vapor chambers
    HYBRID: Combination of multiple cooling methods
    """
    AIR = "air"
    LIQUID = "liquid"
    EVAPORATIVE = "evaporative"
    VAPOR_CHAMBER = "vapor_chamber"
    HYBRID = "hybrid"


@dataclass
class ThermalLayer:
    """
    Represents a single thermal layer in the GPU package stack
    
    Each layer has:
    - geometry: Physical dimensions (thickness, area, shape)
    - material: Thermal and physical properties (conductivity, density, etc.)
    
    This class calculates thermal resistance and capacitance for the layer
    """
    geometry: LayerInfo  # Layer dimensions and geometry info
    material: MaterialProperties  # Material thermal and physical properties
    
    @property
    def thermal_resistance(self) -> float:
        """
        Calculate steady-state conduction resistance through layer
        
        Uses Fourier's Law: R = L / (k * A)
        Where:
        - L = thickness [m]
        - k = thermal conductivity [W/m·K] 
        - A = cross-sectional area [m²]
        
        Returns: Thermal resistance [K/W]
        """
        # R_thermal = thickness / (conductivity * area)
        resistance = self.geometry.thickness / (self.material.k * self.geometry.area)
        print(f"  Layer '{self.geometry.name}': R = {self.geometry.thickness:.2e}m / "
              f"({self.material.k:.1f} W/m·K × {self.geometry.area:.2e}m²) = {resistance:.6f} K/W")
        return resistance
    
    def thermal_resistance_at_temp(self, T: float) -> float:
        """
        Calculate temperature-dependent thermal resistance
        
        Many materials have thermal conductivity that varies with temperature
        k(T) = k₀ × (T₀/T)^n for semiconductors
        
        Args:
            T: Temperature [K]
        Returns: Temperature-dependent thermal resistance [K/W]
        """
        # Get temperature-dependent thermal conductivity
        k_T = self.material.k_temp_dependent(T)
        resistance = self.geometry.thickness / (k_T * self.geometry.area)
        print(f"  Layer '{self.geometry.name}' at {T:.1f}K: k = {k_T:.1f} W/m·K, "
              f"R = {resistance:.6f} K/W")
        return resistance
    
    @property
    def thermal_capacitance(self) -> float:
        """
        Calculate thermal capacitance of layer for transient analysis
        
        Thermal capacitance determines how much energy is needed to change temperature
        C_th = ρ × V × cp
        Where:
        - ρ = density [kg/m³]
        - V = volume [m³] 
        - cp = specific heat capacity [J/kg·K]
        
        Returns: Thermal capacitance [J/K]
        """
        volume = self.geometry.thickness * self.geometry.area  # Volume [m³]
        capacitance = self.material.rho * volume * self.material.cp
        print(f"  Layer '{self.geometry.name}' thermal capacitance: "
              f"{self.material.rho:.0f} kg/m³ × {volume:.2e}m³ × {self.material.cp:.0f} J/kg·K = "
              f"{capacitance:.2e} J/K")
        return capacitance


class ThermalResistance:
    """
    Static methods for calculating various types of thermal resistance
    
    This class implements different heat transfer mechanisms:
    - Convection: Heat transfer to/from fluids
    - Contact: Resistance at material interfaces
    - Spreading: Resistance due to area mismatch (heat spreading)
    """
    
    @staticmethod
    def convection(h: float, area: float) -> float:
        """
        Calculate convection thermal resistance
        
        Based on Newton's Law of Cooling: q = h × A × ΔT
        Therefore: R_conv = 1 / (h × A)
        
        Args:
            h: Heat transfer coefficient [W/m²·K]
            area: Heat transfer surface area [m²]
        Returns: Convection resistance [K/W]
        """
        resistance = 1 / (h * area)
        print(f"  Convection resistance: 1 / ({h:.0f} W/m²·K × {area:.2e}m²) = {resistance:.6f} K/W")
        return resistance
    
    @staticmethod
    def contact(material: MaterialProperties, area: float, 
                pressure: float = 1e6) -> float:
        """
        Calculate contact resistance at material interfaces
        
        Real surfaces have microscopic roughness. Heat transfer occurs only
        at contact points (asperities), creating constriction resistance.
        
        Args:
            material: Material properties of interface
            area: Contact area [m²]
            pressure: Contact pressure [Pa] (affects contact quality)
        Returns: Contact resistance [K/W]
        """
        if material.contact_resistance:
            # Use material-specific contact resistance if available
            resistance = material.contact_resistance / area
            print(f"  Contact resistance (material-specific): "
                  f"{material.contact_resistance:.2e} K·m²/W / {area:.2e}m² = {resistance:.6f} K/W")
        else:
            # Default model based on typical interface properties
            resistance = 1e-5 / area  # Simplified default [K·m²/W]
            print(f"  Contact resistance (default model): "
                  f"1.0e-5 K·m²/W / {area:.2e}m² = {resistance:.6f} K/W")
        return resistance
    
    @staticmethod
    def spreading(k: float, source_area: float, sink_area: float, 
                  thickness: float = 0) -> float:
        """
        Calculate spreading resistance for area mismatch
        
        When heat flows from a small source to larger sink, heat flux lines
        must spread out, creating additional resistance beyond 1D conduction.
        
        Uses Lee et al. correlation for circular geometries.
        
        Args:
            k: Thermal conductivity [W/m·K]
            source_area: Heat source area [m²]
            sink_area: Heat sink area [m²]  
            thickness: Layer thickness [m]
        Returns: Spreading resistance [K/W]
        """
        if sink_area <= source_area:
            print("  No spreading resistance (sink ≤ source area)")
            return 0
        
        # Convert areas to equivalent circular radii
        r_source = np.sqrt(source_area / np.pi)  # Source radius [m]
        r_sink = np.sqrt(sink_area / np.pi)      # Sink radius [m]
        
        print(f"  Spreading analysis: source radius = {r_source*1000:.2f}mm, "
              f"sink radius = {r_sink*1000:.2f}mm")
        
        # Lee et al. correlation parameters
        epsilon = r_source / r_sink  # Radius ratio [-]
        tau = thickness / r_sink if thickness > 0 else 0  # Thickness parameter [-]
        
        # Spreading factor calculation
        psi = np.tanh(1.0 * tau) + epsilon / np.cosh(1.0 * tau)
        
        # Spreading resistance
        resistance = psi / (2 * np.pi * k * r_source)
        
        print(f"  Spreading parameters: ε = {epsilon:.3f}, τ = {tau:.3f}, ψ = {psi:.3f}")
        print(f"  Spreading resistance: {psi:.3f} / (2π × {k:.1f} × {r_source:.2e}) = {resistance:.6f} K/W")
        
        return resistance


class PowerModel:
    """
    GPU power dissipation model including dynamic and leakage components
    
    Total power = Dynamic power + Leakage power
    - Dynamic: Power from switching transistors (clock-dependent)
    - Leakage: Static power from transistor leakage currents (temperature-dependent)
    """
    
    def __init__(self, V_dd: float, f_clock: float, C_eff: float, 
                 alpha: float = 0.7, cores: int = 1):
        """
        Initialize GPU power model
        
        Args:
            V_dd: Supply voltage [V]
            f_clock: Clock frequency [Hz]
            C_eff: Effective switched capacitance [F]
            alpha: Activity factor (fraction of circuits switching) [-]
            cores: Number of processing cores
        """
        self.V_dd = V_dd      # Supply voltage [V]
        self.f_clock = f_clock  # Clock frequency [Hz]  
        self.C_eff = C_eff    # Effective capacitance [F]
        self.alpha = alpha    # Activity factor [-]
        self.cores = cores    # Number of cores
        
        print(f"Power Model Initialized:")
        print(f"  Supply voltage: {V_dd:.2f} V")
        print(f"  Clock frequency: {f_clock/1e9:.2f} GHz")
        print(f"  Effective capacitance: {C_eff*1e12:.1f} pF")
        print(f"  Activity factor: {alpha:.1%}")
        print(f"  Number of cores: {cores}")
        
    def dynamic_power(self) -> float:
        """
        Calculate dynamic power dissipation from CMOS switching
        
        Dynamic power comes from charging/discharging node capacitances
        P_dynamic = α × C_eff × V_dd² × f_clock × N_cores
        
        This power scales quadratically with voltage and linearly with frequency
        
        Returns: Dynamic power [W]
        """
        power = self.cores * self.alpha * self.C_eff * self.V_dd**2 * self.f_clock
        print(f"Dynamic Power Calculation:")
        print(f"  P_dyn = {self.cores} × {self.alpha:.2f} × {self.C_eff:.2e}F × "
              f"{self.V_dd:.2f}²V² × {self.f_clock:.2e}Hz = {power:.2f} W")
        return power
    
    def leakage_power(self, T: float, I_leak_ref: float = 0.1, 
                      T_ref: float = 300, E_a: float = 0.5) -> float:
        """
        Calculate temperature-dependent leakage power
        
        Leakage currents increase exponentially with temperature due to:
        - Subthreshold leakage (below threshold voltage)
        - Gate leakage (through thin gate oxides)
        
        Follows Arrhenius relationship: I ∝ exp(-E_a/k_B×T)
        
        Args:
            T: Junction temperature [K]
            I_leak_ref: Reference leakage current [A]
            T_ref: Reference temperature [K] 
            E_a: Activation energy [eV]
        Returns: Leakage power [W]
        """
        k_B = 8.617e-5  # Boltzmann constant [eV/K]
        
        # Temperature-dependent leakage current
        # I_leak = I_ref × (T/T_ref)² × exp(-E_a/k_B × (1/T - 1/T_ref))
        temp_factor = (T/T_ref)**2
        exp_factor = np.exp(-E_a/k_B * (1/T - 1/T_ref))
        I_leak = I_leak_ref * temp_factor * exp_factor
        
        # Total leakage power
        power = self.cores * I_leak * self.V_dd
        
        print(f"Leakage Power Calculation at T = {T:.1f}K ({T-273.15:.1f}°C):")
        print(f"  Temperature factor: ({T:.1f}/{T_ref:.1f})² = {temp_factor:.3f}")
        print(f"  Exponential factor: exp(-{E_a:.1f}eV / ({k_B:.2e}eV/K × {T:.1f}K)) = {exp_factor:.3f}")
        print(f"  Leakage current: {I_leak_ref:.2f}A × {temp_factor:.3f} × {exp_factor:.3f} = {I_leak:.4f}A")
        print(f"  P_leak = {self.cores} × {I_leak:.4f}A × {self.V_dd:.2f}V = {power:.2f} W")
        
        return power
    
    def total_power(self, T: float) -> float:
        """
        Calculate total power including temperature effects
        
        Total power = Dynamic power + Leakage power(T)
        Creates thermal-electrical coupling since temperature affects power
        
        Args:
            T: Junction temperature [K]
        Returns: Total power dissipation [W]
        """
        P_dyn = self.dynamic_power()
        P_leak = self.leakage_power(T)
        P_total = P_dyn + P_leak
        
        print(f"Total Power: {P_dyn:.2f}W (dynamic) + {P_leak:.2f}W (leakage) = {P_total:.2f}W")
        return P_total


class CoolingModel:
    """
    Base class for cooling models
    
    Cooling models calculate heat transfer coefficients for different
    cooling mechanisms (convection, boiling, etc.)
    """
    
    def __init__(self, fluid_props: FluidProperties):
        """
        Initialize cooling model with fluid properties
        
        Args:
            fluid_props: Thermophysical properties of cooling fluid
        """
        self.fluid = fluid_props
        print(f"Cooling model initialized with fluid: {fluid_props}")
    
    def calculate_htc(self, heat_flux: float, delta_T: float) -> float:
        """
        Calculate heat transfer coefficient
        
        Args:
            heat_flux: Heat flux [W/m²]
            delta_T: Temperature difference [K]
        Returns: Heat transfer coefficient [W/m²·K]
        """
        raise NotImplementedError


class EvaporativeCooling(CoolingModel):
    """
    Evaporative/boiling cooling model for two-phase heat transfer
    
    Boiling heat transfer provides very high heat transfer coefficients
    by utilizing latent heat of vaporization. Includes:
    - Nucleate boiling (most efficient regime)
    - Critical heat flux limit
    - Film boiling (poor heat transfer)
    
    Supports both traditional fluids (water, dielectric) and refrigerants via REFPROP
    """
    
    def __init__(self, fluid_props: Union[FluidProperties, 'RefrigerantProperties']): # type: ignore
        """
        Initialize evaporative cooling model
        
        Args:
            fluid_props: Fluid properties (from data puller or REFPROP)
        """
        self.fluid = fluid_props
        # Check if this is a refrigerant from REFPROP
        self.is_refrigerant = (REFPROP_AVAILABLE and 
                              RefrigerantProperties is not None and 
                              isinstance(fluid_props, RefrigerantProperties))
        print(f"Evaporative cooling model initialized with fluid: {getattr(fluid_props, 'name', 'unknown')}")
        if self.is_refrigerant:
            print(f"  Using REFPROP data for refrigerant calculations")
            print(f"  System pressure: {fluid_props.P/1000:.1f} kPa")
            print(f"  Saturation temperature: {fluid_props.T_sat:.1f}K ({fluid_props.T_sat-273.15:.1f}°C)")
    
    def nucleate_boiling_htc(self, q_flux: float, C_sf: float = None) -> float:
        """
        Calculate nucleate boiling heat transfer coefficient
        
        Uses appropriate correlation based on fluid type:
        - Rohsenow correlation for traditional fluids
        - Cooper correlation for refrigerants (pressure-dependent)
        
        Args:
            q_flux: Heat flux [W/m²]
            C_sf: Surface-fluid combination constant [-] (for Rohsenow)
        Returns: Heat transfer coefficient [W/m²·K]
        """
        print(f"Nucleate Boiling HTC Calculation:")
        print(f"  Heat flux: {q_flux:.0f} W/m²")
        
        if self.is_refrigerant and hasattr(self.fluid, 'P_r'):
            # Use Cooper correlation for refrigerants
            # h = 55 * P_r^0.12 * (-log10(P_r))^-0.55 * M^-0.5 * q^0.67
            
            P_r = self.fluid.P / self.fluid.P_crit  # Reduced pressure
            
            # Estimate molecular weight based on refrigerant (simplified)
            M_weights = {
                'r134a': 102.03,
                'r1234yf': 114.04,
                'r245fa': 134.05,
                'r410a': 72.58,
                'r32': 52.02,
                'water': 18.015,
                'co2': 44.01,
                'ammonia': 17.03
            }
            M = M_weights.get(self.fluid.name, 100)  # Default molecular weight
            
            # Cooper correlation
            h_nb = 55 * P_r**0.12 * (-np.log10(P_r))**-0.55 * M**-0.5 * q_flux**0.67
            
            print(f"  Using Cooper correlation for refrigerant")
            print(f"  Reduced pressure P_r: {P_r:.3f}")
            print(f"  Molecular weight: {M:.1f} g/mol")
            print(f"  Nucleate boiling HTC: {h_nb:.0f} W/m²·K")
            
        else:
            # Use Rohsenow correlation for traditional fluids
            g = 9.81  # Gravitational acceleration [m/s²]
            
            if C_sf is None:
                # Default C_sf values for common surface-fluid combinations
                C_sf_values = {
                    'water': 0.013,      # Water on copper
                    'dielectric_fluid': 0.006,  # Fluorinert on copper
                }
                fluid_name = getattr(self.fluid, 'name', 'unknown').lower()
                C_sf = C_sf_values.get(fluid_name, 0.013)
            
            print(f"  Using Rohsenow correlation")
            print(f"  Surface-fluid constant C_sf: {C_sf:.3f}")
            
            # Rohsenow correlation for wall superheat
            delta_T_sat = (q_flux / (self.fluid.mu_l * self.fluid.h_fg) * 
                          (self.fluid.sigma / (g * (self.fluid.rho_l - self.fluid.rho_v)))**0.5 *
                          (C_sf * self.fluid.h_fg * self.fluid.Pr_l**1.7 / self.fluid.cp_l)**(1/3))
            
            # Heat transfer coefficient
            h_nb = q_flux / delta_T_sat
            
            print(f"  Wall superheat: {delta_T_sat:.2f} K")
            print(f"  Nucleate boiling HTC: {h_nb:.0f} W/m²·K")
        
        return h_nb
    
    def critical_heat_flux(self) -> float:
        """
        Calculate critical heat flux limit
        
        CHF is the maximum heat flux before transition to film boiling
        Uses appropriate correlation based on fluid type
        
        Returns: Critical heat flux [W/m²]
        """
        g = 9.81  # Gravitational acceleration [m/s²]
        
        if self.is_refrigerant and hasattr(self.fluid, 'P_r'):
            # Use Kutateladze-Zuber correlation with pressure correction for refrigerants
            P_r = self.fluid.P / self.fluid.P_crit
            
            # Pressure correction factor (Mostinski)
            K_p = 0.374 * (1 + 4 * P_r * (1 + 0.222 * P_r**2))
            
            # Modified Zuber correlation
            q_chf = K_p * 0.131 * self.fluid.h_fg * self.fluid.rho_v**0.5 * \
                    (g * self.fluid.sigma * (self.fluid.rho_l - self.fluid.rho_v))**0.25
            
            print(f"Critical Heat Flux Calculation (Refrigerant):")
            print(f"  Reduced pressure P_r: {P_r:.3f}")
            print(f"  Pressure correction K_p: {K_p:.3f}")
            
        else:
            # Standard Zuber correlation for traditional fluids
            q_chf = 0.131 * self.fluid.h_fg * self.fluid.rho_v**0.5 * \
                    (g * self.fluid.sigma * (self.fluid.rho_l - self.fluid.rho_v))**0.25
            
            print(f"Critical Heat Flux Calculation (Standard):")
        
        print(f"  Latent heat: {self.fluid.h_fg:.0f} J/kg")
        print(f"  Vapor density: {self.fluid.rho_v:.2f} kg/m³")
        print(f"  Surface tension: {self.fluid.sigma:.4f} N/m")
        print(f"  Density difference: {self.fluid.rho_l - self.fluid.rho_v:.1f} kg/m³")
        print(f"  Critical heat flux: {q_chf:.0f} W/m²")
        
        return q_chf
    
    def film_boiling_htc(self, delta_T: float, L_char: float = 0.01) -> float:
        """
        Calculate film boiling heat transfer coefficient
        
        Film boiling occurs beyond CHF when vapor film blankets surface
        Uses Bromley correlation for horizontal surfaces
        
        Args:
            delta_T: Wall superheat [K]
            L_char: Characteristic length [m]
        Returns: Film boiling HTC [W/m²·K]
        """
        g = 9.81  # Gravitational acceleration [m/s²]
        
        # Use vapor properties for film boiling
        if self.is_refrigerant:
            k_v = self.fluid.k_v
            mu_v = self.fluid.mu_v
            cp_v = self.fluid.cp_v
        else:
            # For traditional fluids, estimate vapor properties if not available
            k_v = getattr(self.fluid, 'k_v', self.fluid.k_l * 0.1)  # Rough estimate
            mu_v = getattr(self.fluid, 'mu_v', self.fluid.mu_l * 0.01)
            cp_v = getattr(self.fluid, 'cp_v', self.fluid.cp_l * 0.5)
        
        # Modified latent heat including sensible heat
        h_fg_eff = self.fluid.h_fg + 0.4 * cp_v * delta_T
        
        # Bromley correlation for film boiling
        h_fb = 0.62 * ((k_v**3 * self.fluid.rho_v * 
                       (self.fluid.rho_l - self.fluid.rho_v) * g * h_fg_eff) /
                      (mu_v * L_char * delta_T))**0.25
        
        print(f"Film Boiling HTC Calculation:")
        print(f"  Wall superheat: {delta_T:.2f} K")
        print(f"  Characteristic length: {L_char:.3f} m")
        print(f"  Effective latent heat: {h_fg_eff/1e6:.2f} MJ/kg")
        print(f"  Film boiling HTC: {h_fb:.0f} W/m²·K")
        
        return h_fb


class VaporChamberModel:
    """
    Vapor chamber thermal model for advanced cooling
    
    Vapor chambers use two-phase heat transfer:
    1. Evaporation at hot spot (absorbs latent heat)
    2. Vapor transport (low resistance)
    3. Condensation at cold regions (releases heat)
    4. Liquid return via capillary pumping
    """
    
    def __init__(self, data_puller: DataPuller, config_key: str):
        """
        Initialize vapor chamber model
        
        Args:
            data_puller: Data interface for material/fluid properties
            config_key: Configuration identifier for vapor chamber design
        """
        self.data = data_puller
        self.config = data_puller.get_vapor_chamber_config(config_key)
        self.geometry = self.config['geometry']
        self.wick = self.config['wick_properties']
        
        print(f"Vapor Chamber Model Initialized:")
        print(f"  Configuration: {config_key}")
        print(f"  Evaporator area: {self.geometry['evaporator_area']:.2e} m²")
        print(f"  Condenser area: {self.geometry['condenser_area']:.2e} m²")
        print(f"  Wick thermal conductivity: {self.wick['thermal_conductivity']:.1f} W/m·K")
    
    def effective_conductivity(self) -> float:
        """
        Calculate effective thermal conductivity
        
        Vapor chambers have very high effective conductivity due to
        latent heat transport. Can be 10-100× higher than copper.
        
        Returns: Effective thermal conductivity [W/m·K]
        """
        k_eff = 10000  # Very high effective conductivity for vapor chambers
        print(f"  Vapor chamber effective conductivity: {k_eff:.0f} W/m·K")
        return k_eff
    
    def thermal_resistance(self, fluid_key: str = 'water') -> float:
        """
        Calculate total thermal resistance of vapor chamber
        
        Includes:
        - Evaporation resistance
        - Vapor flow resistance  
        - Condensation resistance
        - Wick conduction resistances
        
        Args:
            fluid_key: Working fluid identifier
        Returns: Total thermal resistance [K/W]
        """
        fluid = self.data.get_fluid_properties(fluid_key)
        
        print(f"Vapor Chamber Thermal Resistance Analysis:")
        print(f"  Working fluid: {fluid_key}")
        
        # Component areas
        A_evap = self.geometry['evaporator_area']  # Evaporator area [m²]
        A_cond = self.geometry['condenser_area']   # Condenser area [m²]
        
        print(f"  Evaporator area: {A_evap:.2e} m²")
        print(f"  Condenser area: {A_cond:.2e} m²")
        
        # Component resistances (simplified model)
        R_evap = 1 / (20000 * A_evap)  # Evaporation resistance [K/W]
        R_cond = 1 / (20000 * A_cond)  # Condensation resistance [K/W]
        R_vapor = 0.0001               # Vapor flow resistance [K/W] (very small)
        
        print(f"  Evaporation resistance: {R_evap:.6f} K/W")
        print(f"  Condensation resistance: {R_cond:.6f} K/W")
        print(f"  Vapor flow resistance: {R_vapor:.6f} K/W")
        
        # Wick conduction resistances
        wick_thickness = self.geometry['wick_thickness']  # Wick thickness [m]
        k_wick = self.wick['thermal_conductivity']        # Wick conductivity [W/m·K]
        
        R_wick_evap = wick_thickness / (k_wick * A_evap)  # Evaporator wick resistance [K/W]
        R_wick_cond = wick_thickness / (k_wick * A_cond)  # Condenser wick resistance [K/W]
        
        print(f"  Wick evaporator resistance: {R_wick_evap:.6f} K/W")
        print(f"  Wick condenser resistance: {R_wick_cond:.6f} K/W")
        
        # Total resistance
        R_total = R_evap + R_vapor + R_cond + R_wick_evap + R_wick_cond
        print(f"  Total vapor chamber resistance: {R_total:.6f} K/W")
        
        return R_total
    
    def capillary_limit(self, fluid_key: str = 'water') -> float:
        """
        Calculate maximum heat transfer limited by capillary pumping
        
        The wick must provide enough capillary pressure to overcome:
        - Viscous pressure drop in liquid flow
        - Vapor pressure drop
        - Gravitational head
        
        Args:
            fluid_key: Working fluid identifier
        Returns: Maximum heat transfer [W]
        """
        fluid = self.data.get_fluid_properties(fluid_key)
        
        print(f"Capillary Limit Analysis:")
        
        # Maximum capillary pressure from Young-Laplace equation
        pore_radius = self.wick['pore_radius']  # Wick pore radius [m]
        delta_P_cap = 2 * fluid.sigma / pore_radius  # Maximum capillary pressure [Pa]
        
        print(f"  Pore radius: {pore_radius*1e6:.1f} μm")
        print(f"  Surface tension: {fluid.sigma:.4f} N/m")
        print(f"  Maximum capillary pressure: {delta_P_cap:.0f} Pa")
        
        # Simplified capillary limit calculation
        # Actual calculation would include detailed flow analysis
        permeability = self.wick['permeability']      # Wick permeability [m²]
        eff_length = self.geometry['effective_length'] # Effective flow length [m]
        wick_area = self.geometry['wick_area']        # Wick cross-sectional area [m²]
        
        Q_max = delta_P_cap * permeability * wick_area * fluid.rho_l * fluid.h_fg / \
                (fluid.mu_l * eff_length)
        
        print(f"  Wick permeability: {permeability:.2e} m²")
        print(f"  Effective length: {eff_length:.3f} m")
        print(f"  Maximum heat transfer: {Q_max:.1f} W")
        
        return Q_max


class JunctionToSurfaceCalculator:
    """
    Main thermal calculator class for GPU junction-to-surface analysis
    
    This class:
    1. Manages thermal layer stack configuration
    2. Calculates steady-state temperature distribution
    3. Solves nonlinear thermal-electrical coupling
    4. Provides visualization and sensitivity analysis
    """
    
    def __init__(self, data_puller: Optional[DataPuller] = None):
        """
        Initialize thermal calculator
        
        Args:
            data_puller: Data interface for materials and configurations
        """
        self.data = data_puller or DataPuller()
        self.layers: List[ThermalLayer] = []
        self.config_name: str = ""
        print("Junction-to-Surface Thermal Calculator Initialized")
    
    def load_configuration(self, config_key: str):
        """
        Load a predefined chip configuration from database
        
        Loads complete thermal stack including:
        - Die layer (silicon)
        - Thermal interface material (TIM)
        - Integrated heat spreader (IHS)
        - Additional packaging layers
        
        Args:
            config_key: Configuration identifier (e.g., 'standard_gpu')
        """
        print(f"\nLoading Configuration: {config_key}")
        print("=" * 40)
        
        layer_infos = self.data.get_chip_configuration(config_key)
        self.config_name = config_key
        self.layers = []
        
        print(f"Configuration loaded with {len(layer_infos)} layers:")
        
        for i, layer_info in enumerate(layer_infos):
            material = self.data.get_material(layer_info.material_key)
            self.layers.append(ThermalLayer(layer_info, material))
            
            print(f"  Layer {i+1}: {layer_info.name}")
            print(f"    Material: {layer_info.material_key}")
            print(f"    Thickness: {layer_info.thickness*1000:.3f} mm")
            print(f"    Area: {layer_info.area*1e6:.1f} mm²")
            print(f"    Thermal conductivity: {material.k:.1f} W/m·K")
    
    def add_custom_layer(self, name: str, thickness: float, area: float, 
                        material_key: str):
        """
        Add a custom layer to the thermal stack
        
        Args:
            name: Layer name/description
            thickness: Layer thickness [m]
            area: Cross-sectional area [m²]
            material_key: Material identifier
        """
        print(f"\nAdding Custom Layer: {name}")
        print(f"  Thickness: {thickness*1000:.3f} mm")
        print(f"  Area: {area*1e6:.1f} mm²")
        print(f"  Material: {material_key}")
        
        layer_info = LayerInfo(name, thickness, area, material_key)
        material = self.data.get_material(material_key)
        self.layers.append(ThermalLayer(layer_info, material))
    
    def calculate_steady_state(self, power: float, T_ambient: float,
                             cooling_type: CoolingType,
                             cooling_params: Dict) -> Dict[str, float]:
        """
        Calculate steady-state temperature distribution through thermal stack
        
        Solves thermal resistance network:
        T_junction = T_ambient + Q × R_total
        
        Args:
            power: Heat dissipation [W]
            T_ambient: Ambient temperature [K]
            cooling_type: Type of cooling system
            cooling_params: Cooling system parameters
            
        Returns: Dictionary with temperatures, resistances, and results
        """
        print(f"\n" + "="*60)
        print(f"STEADY-STATE THERMAL ANALYSIS")
        print(f"="*60)
        print(f"Power dissipation: {power:.2f} W")
        print(f"Ambient temperature: {T_ambient:.1f} K ({T_ambient-273.15:.1f}°C)")
        print(f"Cooling type: {cooling_type.value}")
        
        # Initialize results dictionary
        results = {
            'power': power,
            'T_ambient': T_ambient,
            'cooling_type': cooling_type.value,
            'resistances': {},
            'temperatures': {}
        }
        
        print(f"\nTHERMAL RESISTANCE CALCULATIONS:")
        print(f"-" * 40)
        
        # Calculate conduction resistances through each layer
        print("1. Layer Conduction Resistances:")
        R_total = 0
        R_layers = []
        
        for i, layer in enumerate(self.layers):
            print(f"\n  Layer {i+1}: {layer.geometry.name}")
            R_layer = layer.thermal_resistance
            R_layers.append(R_layer)
            R_total += R_layer
            results['resistances'][f'R_{layer.geometry.name}'] = R_layer
        
        print(f"\n  Total conduction resistance: {R_total:.6f} K/W")
        
        # Calculate spreading resistances for area mismatches
        print("\n2. Spreading Resistances:")
        spreading_count = 0
        for i in range(len(self.layers) - 1):
            if self.layers[i].geometry.area < self.layers[i+1].geometry.area:
                print(f"\n  Spreading from {self.layers[i].geometry.name} to {self.layers[i+1].geometry.name}:")
                R_spread = ThermalResistance.spreading(
                    self.layers[i+1].material.k,
                    self.layers[i].geometry.area,
                    self.layers[i+1].geometry.area,
                    self.layers[i+1].geometry.thickness
                )
                R_total += R_spread
                results['resistances'][f'R_spread_{i}_{i+1}'] = R_spread
                spreading_count += 1
        
        if spreading_count == 0:
            print("  No spreading resistances (no area mismatches)")
        
        # Calculate contact resistances at interfaces  
        print("\n3. Contact Resistances:")
        for i in range(len(self.layers) - 1):
            print(f"\n  Contact between {self.layers[i].geometry.name} and {self.layers[i+1].geometry.name}:")
            contact_area = min(self.layers[i].geometry.area, self.layers[i+1].geometry.area)
            R_contact = ThermalResistance.contact(self.layers[i].material, contact_area)
            R_total += R_contact
            results['resistances'][f'R_contact_{i}_{i+1}'] = R_contact
        
        print(f"\n  Package total resistance: {R_total:.6f} K/W")
        
        # Calculate cooling system resistance and final temperatures
        print(f"\nCOOLING SYSTEM ANALYSIS:")
        print(f"-" * 40)
        
        if cooling_type == CoolingType.AIR:
            print("Air Cooling Analysis:")
            h_air = cooling_params.get('h_air', 50)    # Heat transfer coefficient [W/m²·K]
            A_sink = cooling_params.get('A_sink', 0.01)  # Heat sink area [m²]
            
            print(f"  Heat transfer coefficient: {h_air:.0f} W/m²·K")
            print(f"  Heat sink area: {A_sink*1e4:.1f} cm²")
            
            R_cooling = ThermalResistance.convection(h_air, A_sink)
            
            # Temperature calculations
            T_junction = T_ambient + power * (R_total + R_cooling)
            T_case = T_ambient + power * R_cooling
            
        elif cooling_type == CoolingType.LIQUID:
            print("Liquid Cooling Analysis:")
            h_liquid = cooling_params.get('h_liquid', 5000)  # HTC [W/m²·K]
            A_cold = cooling_params.get('A_cold', self.layers[-1].geometry.area)  # Cold plate area [m²]
            T_coolant = cooling_params.get('T_coolant', T_ambient)  # Coolant temperature [K]
            
            print(f"  Heat transfer coefficient: {h_liquid:.0f} W/m²·K")
            print(f"  Cold plate area: {A_cold*1e4:.1f} cm²")
            print(f"  Coolant temperature: {T_coolant:.1f} K ({T_coolant-273.15:.1f}°C)")
            
            R_cooling = ThermalResistance.convection(h_liquid, A_cold)
            
            # Temperature calculations (referenced to coolant, not ambient)
            T_junction = T_coolant + power * (R_total + R_cooling)
            T_case = T_coolant + power * R_cooling
            
        elif cooling_type == CoolingType.EVAPORATIVE:
            print("Evaporative Cooling Analysis:")
            
            # Get fluid type and system parameters
            fluid_key = cooling_params.get('fluid', 'water')
            system_pressure = cooling_params.get('pressure', 101325)  # Default 1 atm
            use_refprop = cooling_params.get('use_refprop', False)
            
            # Determine if this is a refrigerant that needs REFPROP
            refrigerants = ['r134a', 'r1234yf', 'r1234ze', 'r245fa', 'r410a', 'r32', 
                           'r404a', 'r507a', 'r22', 'r744', 'r717', 'r290', 'r600a']
            is_refrigerant = fluid_key.lower() in refrigerants
            
            # Get fluid properties
            if is_refrigerant and REFPROP_AVAILABLE and (use_refprop or is_refrigerant):
                print(f"  Using REFPROP for refrigerant: {fluid_key}")
                print(f"  System pressure: {system_pressure/1000:.1f} kPa")
                
                # Initialize REFPROP interface
                refprop = RefpropInterface()
                refprop.set_fluid(fluid_key)
                
                # Get saturation properties at system pressure
                fluid = refprop.get_saturation_properties(P=system_pressure)
                evap_cooling = EvaporativeCooling(fluid)
                
            else:
                # Use traditional fluid properties from data puller
                print(f"  Working fluid: {fluid_key}")
                
                try:
                    fluid = self.data.get_fluid_properties(fluid_key)
                except ValueError:
                    print(f"  Warning: Fluid '{fluid_key}' not found in database, using water")
                    fluid = self.data.get_fluid_properties('water')
                
                evap_cooling = EvaporativeCooling(fluid)
                print(f"  Saturation temperature: {fluid.T_sat:.1f} K ({fluid.T_sat-273.15:.1f}°C)")
            
            # Calculate heat flux and check against CHF
            surface_area = self.layers[-1].geometry.area
            q_flux = power / surface_area
            q_chf = evap_cooling.critical_heat_flux()
            chf_ratio = q_flux / q_chf
            
            print(f"  Heat flux: {q_flux:.0f} W/m²")
            print(f"  Critical heat flux: {q_chf:.0f} W/m²")
            print(f"  CHF ratio: {chf_ratio:.1%}")
            
            if chf_ratio > 0.8:
                print(f"  ⚠️  WARNING: Operating near CHF limit ({chf_ratio:.1%} of CHF)")
            
            # Calculate boiling heat transfer coefficient
            if chf_ratio < 1.0:
                # Nucleate boiling regime
                h_evap = evap_cooling.nucleate_boiling_htc(q_flux)
            else:
                # Beyond CHF - use film boiling
                print(f"  ⚠️  WARNING: Heat flux exceeds CHF! Using film boiling correlation")
                # Estimate wall superheat for film boiling
                delta_T_est = 150  # Initial estimate [K]
                h_evap = evap_cooling.film_boiling_htc(delta_T_est)
            
            R_cooling = 1 / (h_evap * surface_area)
            
            # Temperature calculations (referenced to saturation temperature)
            T_case = fluid.T_sat + power * R_cooling
            T_junction = T_case + power * R_total
            
        elif cooling_type == CoolingType.VAPOR_CHAMBER:
            print("Vapor Chamber Cooling Analysis:")
            vc_config = cooling_params.get('vc_config', 'standard_vc')
            vc_model = VaporChamberModel(self.data, vc_config)
            
            # Calculate vapor chamber resistance
            R_vc = vc_model.thermal_resistance()
            T_condenser = cooling_params.get('T_condenser', T_ambient)
            
            print(f"  Condenser temperature: {T_condenser:.1f} K ({T_condenser-273.15:.1f}°C)")
            
            # Temperature calculations
            T_case = T_condenser + power * R_vc
            T_junction = T_case + power * R_total
            R_cooling = R_vc
            
        else:
            raise ValueError(f"Unsupported cooling type: {cooling_type}")
        
        # Store final results
        results['resistances']['R_cooling'] = R_cooling
        results['resistances']['R_total'] = R_total + R_cooling
        results['temperatures']['T_junction'] = T_junction
        results['temperatures']['T_case'] = T_case
        
        print(f"\nFINAL TEMPERATURE RESULTS:")
        print(f"-" * 40)
        print(f"Junction temperature: {T_junction:.1f} K ({T_junction-273.15:.1f}°C)")
        print(f"Case temperature: {T_case:.1f} K ({T_case-273.15:.1f}°C)")
        print(f"Total thermal resistance: {R_total + R_cooling:.6f} K/W")
        print(f"Temperature rise: {T_junction - T_ambient:.1f} K")
        
        # Calculate intermediate temperatures through stack
        print(f"\nTEMPERATURE DISTRIBUTION:")
        print(f"-" * 40)
        T_current = T_junction
        print(f"Junction: {T_current:.1f} K ({T_current-273.15:.1f}°C)")
        
        for i, (layer, R) in enumerate(zip(self.layers, R_layers)):
            T_current -= power * R  # Temperature drops across each resistance
            temp_key = f'T_after_{layer.geometry.name}'
            results['temperatures'][temp_key] = T_current
            print(f"After {layer.geometry.name}: {T_current:.1f} K ({T_current-273.15:.1f}°C)")
        
        return results
    
    def solve_nonlinear(self, power_model: PowerModel, T_ambient: float,
                       cooling_type: CoolingType, cooling_params: Dict,
                       max_iter: int = 500, tol: float = 1) -> Dict[str, float]:
        """
        Solve coupled thermal-electrical problem with iteration
        
        The nonlinearity arises from temperature-dependent leakage power:
        - Power depends on temperature (leakage increases with T)
        - Temperature depends on power (thermal resistance)
        
        Solution uses fixed-point iteration to converge on consistent T and P
        
        Args:
            power_model: GPU power dissipation model
            T_ambient: Ambient temperature [K]
            cooling_type: Cooling system type
            cooling_params: Cooling parameters
            max_iter: Maximum iterations
            tol: Convergence tolerance [K]
            
        Returns: Converged solution with power breakdown
        """
        print(f"\n" + "="*60)
        print(f"NONLINEAR THERMAL-ELECTRICAL COUPLING SOLUTION")
        print(f"="*60)
        print(f"Solving coupled problem with leakage power feedback")
        print(f"Convergence tolerance: {tol:.2f} K")
        print(f"Maximum iterations: {max_iter}")
        
        # Initial temperature guess based on dynamic power only
        P_dyn = power_model.dynamic_power()
        T_j = T_ambient + P_dyn * 0.3  # Conservative initial guess
        
        print(f"\nInitial guess: T_junction = {T_j:.1f} K ({T_j-273.15:.1f}°C)")
        print(f"Dynamic power: {P_dyn:.2f} W")
        
        print(f"\nITERATION PROCESS:")
        print(f"-" * 40)
        
        for iteration in range(max_iter):
            # Calculate total power at current temperature
            power_total = power_model.total_power(T_j)
            
            # Calculate new temperature from thermal analysis
            results = self.calculate_steady_state(
                power_total, T_ambient, cooling_type, cooling_params
            )
            T_j_new = results['temperatures']['T_junction']
            
            # Check convergence
            error = abs(T_j_new - T_j)
            print(f"Iteration {iteration+1:2d}: T = {T_j:.1f}K → {T_j_new:.1f}K, "
                  f"P = {power_total:.2f}W, Error = {error:.3f}K")
            
            if error < tol:
                print(f"\n✅ CONVERGED after {iteration + 1} iterations!")
                results['converged'] = True
                results['iterations'] = iteration + 1
                results['power_dynamic'] = power_model.dynamic_power()
                results['power_leakage'] = power_model.leakage_power(T_j_new)
                
                print(f"\nFINAL CONVERGED SOLUTION:")
                print(f"-" * 40)
                print(f"Junction temperature: {T_j_new:.1f} K ({T_j_new-273.15:.1f}°C)")
                print(f"Total power: {power_total:.2f} W")
                print(f"  Dynamic power: {results['power_dynamic']:.2f} W ({results['power_dynamic']/power_total:.1%})")
                print(f"  Leakage power: {results['power_leakage']:.2f} W ({results['power_leakage']/power_total:.1%})")
                
                return results
            
            T_j = T_j_new  # Update temperature for next iteration
        
        print(f"\n❌ Failed to converge after {max_iter} iterations")
        print(f"Final error: {error:.3f} K")
        results['converged'] = False
        results['iterations'] = max_iter
        return results
    
    def plot_temperature_profile(self, results: Dict[str, float], 
                               save_path: Optional[str] = None):
        """
        Plot temperature distribution through thermal stack
        
        Creates visualization showing:
        - Temperature profile from junction to ambient
        - Layer boundaries and thermal resistances
        - Key temperature points
        
        Args:
            results: Results from thermal analysis
            save_path: Optional path to save plot
        """
        print(f"\nGenerating temperature profile visualization...")
        
        # Build position and temperature arrays
        positions = [0]  # Start at junction (position = 0)
        temperatures = [results['temperatures']['T_junction']]
        labels = ['Junction']
        
        print(f"Temperature profile data points:")
        print(f"  Junction: {temperatures[0]:.1f} K at position {positions[0]:.3f} mm")
        
        current_pos = 0
        for i, layer in enumerate(self.layers):
            current_pos += layer.geometry.thickness * 1000  # Convert to mm
            positions.append(current_pos)
            
            # Get temperature after this layer
            temp_key = f'T_after_{layer.geometry.name}'
            if temp_key in results['temperatures']:
                temp = results['temperatures'][temp_key]
            else:
                temp = results['temperatures']['T_case']
            
            temperatures.append(temp)
            labels.append(layer.geometry.name.upper())
            print(f"  After {layer.geometry.name}: {temp:.1f} K at position {current_pos:.3f} mm")
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot temperature profile
        temp_celsius = np.array(temperatures) - 273.15  # Convert to Celsius
        ax.plot(positions, temp_celsius, 'b-', linewidth=2, label='Temperature Profile')
        ax.scatter(positions, temp_celsius, color='red', s=100, zorder=5)
        
        # Add temperature labels
        for pos, temp, label in zip(positions, temperatures, labels):
            ax.annotate(f'{label}\n{temp-273.15:.1f}°C', 
                       xy=(pos, temp-273.15), 
                       xytext=(pos, temp-273.15+2),
                       ha='center', fontsize=9, weight='bold')
        
        # Formatting
        ax.set_xlabel('Distance from Junction [mm]', fontsize=12)
        ax.set_ylabel('Temperature [°C]', fontsize=12)
        ax.set_title(f'Temperature Profile - {self.config_name} ({results["power"]:.1f}W)', 
                    fontsize=14, weight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.5, positions[-1] + 0.5)
        
        # Add thermal resistance annotations
        y_pos = ax.get_ylim()[0] + 0.1 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        for i, layer in enumerate(self.layers):
            R_key = f'R_{layer.geometry.name}'
            if R_key in results['resistances']:
                R_value = results['resistances'][R_key]
                x_pos = (positions[i] + positions[i+1]) / 2
                ax.text(x_pos, y_pos, f'R={R_value:.3f} K/W', 
                       ha='center', fontsize=8, style='italic', alpha=0.7)
        
        # Add power and cooling info
        info_text = f"Power: {results['power']:.1f}W\nCooling: {results['cooling_type']}\n"
        info_text += f"Total R: {results['resistances']['R_total']:.3f} K/W"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()
        
        return fig, ax
    
    def sensitivity_analysis(self, base_power: float, T_ambient: float,
                           cooling_type: CoolingType, cooling_params: Dict,
                           parameter: str, values: List[float]) -> Dict[str, List[float]]:
        """
        Perform sensitivity analysis on a parameter
        
        Varies one parameter while keeping others constant to understand
        system sensitivity and identify critical design parameters
        
        Args:
            base_power: Baseline power dissipation [W]
            T_ambient: Ambient temperature [K]
            cooling_type: Cooling system type
            cooling_params: Baseline cooling parameters
            parameter: Parameter name to vary
            values: List of parameter values to test
            
        Returns: Dictionary with parameter sweep results
        """
        print(f"\n" + "="*60)
        print(f"SENSITIVITY ANALYSIS")
        print(f"="*60)
        print(f"Parameter: {parameter}")
        print(f"Baseline power: {base_power:.2f} W")
        print(f"Parameter range: {min(values):.3f} to {max(values):.3f}")
        print(f"Number of points: {len(values)}")
        
        results = {
            'parameter_values': values,
            'T_junction': [],
            'T_case': [],
            'R_total': []
        }
        
        print(f"\nParameter sweep results:")
        print(f"-" * 40)
        
        for i, value in enumerate(values):
            # Modify parameter
            if parameter == 'power':
                power = value
                params = cooling_params.copy()
                print(f"Point {i+1:2d}: Power = {power:.2f} W")
            elif parameter in cooling_params:
                power = base_power
                params = cooling_params.copy()
                params[parameter] = value
                print(f"Point {i+1:2d}: {parameter} = {value:.3f}")
            else:
                raise ValueError(f"Unknown parameter: {parameter}")
            
            # Calculate thermal response
            calc_results = self.calculate_steady_state(
                power, T_ambient, cooling_type, params
            )
            
            # Store results
            T_j = calc_results['temperatures']['T_junction']
            T_c = calc_results['temperatures']['T_case']
            R_tot = calc_results['resistances']['R_total']
            
            results['T_junction'].append(T_j)
            results['T_case'].append(T_c)
            results['R_total'].append(R_tot)
            
            print(f"         T_junction = {T_j:.1f}K, T_case = {T_c:.1f}K, R_total = {R_tot:.4f}K/W")
        
        # Calculate sensitivity metrics
        dT_junction = max(results['T_junction']) - min(results['T_junction'])
        dT_case = max(results['T_case']) - min(results['T_case'])
        param_range = max(values) - min(values)
        
        print(f"\nSensitivity Summary:")
        print(f"-" * 40)
        print(f"Junction temperature range: {dT_junction:.2f} K")
        print(f"Case temperature range: {dT_case:.2f} K")
        print(f"Parameter range: {param_range:.3f}")
        print(f"Junction sensitivity: {dT_junction/param_range:.2f} K per unit")
        print(f"Case sensitivity: {dT_case/param_range:.2f} K per unit")
        
        return results


def main():
    """
    Example usage demonstrating thermal analysis capabilities
    
    This example shows:
    1. Loading GPU configuration
    2. Setting up power model
    3. Comparing different cooling methods
    4. Solving nonlinear thermal-electrical coupling
    5. Generating visualization
    """
    print("GPU JUNCTION-TO-SURFACE THERMAL CALCULATOR")
    print("="*60)
    print("Advanced thermal analysis for semiconductor devices")
    print("Includes material properties, heat transfer mechanisms,")
    print("and thermal-electrical coupling effects")
    
    # Create calculator instance
    calc = JunctionToSurfaceCalculator()
    
    # Load standard GPU configuration
    calc.load_configuration('standard_gpu')
    
    # Define GPU power model
    print(f"\nGPU POWER MODEL SETUP:")
    print(f"-" * 40)
    power_model = PowerModel(
        V_dd=1.1,       # Supply voltage [V]
        f_clock=2.0e9,  # Clock frequency [Hz] - 2 GHz
        C_eff=1e-9,     # Effective capacitance [F] - 1 nF
        alpha=0.7,      # Activity factor - 70% switching
        cores=3584      # Number of CUDA cores (example)
    )
    
    print(f"\nCOOLING METHOD COMPARISON:")
    print(f"="*60)
    
    # Define cooling scenarios for comparison
    cooling_scenarios = [
        {
            'name': 'High-Performance Air Cooling',
            'cooling_type': CoolingType.AIR,
            'params': {
                'h_air': 50,      # Heat transfer coefficient [W/m²·K]
                'A_sink': 0.01    # Heat sink area [m²] - 100 cm²
            }
        },
        {
            'name': 'Liquid Cooling (AIO)',
            'cooling_type': CoolingType.LIQUID,
            'params': {
                'h_liquid': 5000,     # Liquid HTC [W/m²·K]
                'A_cold': 0.0016,     # Cold plate area [m²] - 16 cm²
                'T_coolant': 288      # Coolant temperature [K] - 15°C
            }
        },
        {
            'name': 'Vapor Chamber Cooling',
            'cooling_type': CoolingType.VAPOR_CHAMBER,
            'params': {
                'vc_config': 'standard_vc',  # Vapor chamber configuration
                'T_condenser': 298           # Condenser temperature [K] - 25°C
            }
        }
    ]
    
    # Analyze each cooling scenario
    scenario_results = []
    for scenario in cooling_scenarios:
        print(f"\n{scenario['name'].upper()}:")
        print(f"=" * len(scenario['name']))
        
        # Solve nonlinear thermal-electrical problem
        results = calc.solve_nonlinear(
            power_model, 
            T_ambient=298,  # 25°C ambient
            cooling_type=scenario['cooling_type'],
            cooling_params=scenario['params']
        )
        
        scenario_results.append(results)
        
        # Print summary results
        if results['converged']:
            print(f"\n📊 PERFORMANCE SUMMARY:")
            print(f"   Junction Temperature: {results['temperatures']['T_junction']-273.15:.1f}°C")
            print(f"   Case Temperature: {results['temperatures']['T_case']-273.15:.1f}°C")
            print(f"   Total Power: {results['power']:.1f}W")
            print(f"     • Dynamic: {results['power_dynamic']:.1f}W ({results['power_dynamic']/results['power']:.1%})")
            print(f"     • Leakage: {results['power_leakage']:.1f}W ({results['power_leakage']/results['power']:.1%})")
            print(f"   Total Thermal Resistance: {results['resistances']['R_total']:.3f} K/W")
            print(f"   Temperature Rise: {results['temperatures']['T_junction'] - 298:.1f} K")
            
            # Performance assessment
            T_junction_C = results['temperatures']['T_junction'] - 273.15
            if T_junction_C < 85:
                status = "✅ EXCELLENT"
            elif T_junction_C < 95:
                status = "✅ GOOD"
            elif T_junction_C < 105:
                status = "⚠️  MARGINAL"
            else:
                status = "❌ POOR"
            print(f"   Thermal Performance: {status}")
        else:
            print(f"   ❌ Solution did not converge!")
    
    # Generate visualization for best cooling solution
    best_result = min(scenario_results, key=lambda x: x['temperatures']['T_junction'])
    print(f"\nGenerating temperature profile for best cooling solution...")
    calc.plot_temperature_profile(best_result)
    
    # Demonstrate sensitivity analysis
    print(f"\nSENSITIVITY ANALYSIS EXAMPLE:")
    print(f"="*60)
    
    # Analyze sensitivity to liquid cooling HTC
    if any(s['cooling_type'] == CoolingType.LIQUID for s in cooling_scenarios):
        liquid_params = next(s['params'] for s in cooling_scenarios if s['cooling_type'] == CoolingType.LIQUID)
        
        htc_values = np.linspace(1000, 10000, 10)  # Range of HTCs to test
        sensitivity_results = calc.sensitivity_analysis(
            base_power=150.0,  # Fixed power
            T_ambient=298,
            cooling_type=CoolingType.LIQUID,
            cooling_params=liquid_params,
            parameter='h_liquid',
            values=htc_values.tolist()
        )
        
        print(f"Sensitivity analysis complete!")
    
    print(f"\n" + "="*60)
    print(f"THERMAL ANALYSIS COMPLETE")
    print(f"="*60)
    print(f"All calculations completed successfully!")
    print(f"Check the generated plots and results above.")


if __name__ == "__main__":
    main()