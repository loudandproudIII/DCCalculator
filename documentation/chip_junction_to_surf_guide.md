# GPU Junction-to-Surface Thermal Calculator
## Physics and Engineering Reference Guide

This document provides detailed explanations of the thermal physics, heat transfer mechanisms, and engineering correlations implemented in the GPU thermal calculator code.

---

## 1. Core Thermal Concepts

### 1.1 Thermal Resistance Network Model

The calculator uses the **electrical-thermal analogy**, where:
- **Temperature difference (ΔT)** is analogous to voltage
- **Heat flow (Q)** is analogous to current  
- **Thermal resistance (R_th)** is analogous to electrical resistance

The fundamental equation: **Q = ΔT / R_th**

This allows complex thermal systems to be analyzed as resistor networks, with resistances in series adding directly and parallel resistances following the reciprocal rule.

### 1.2 Heat Transfer Mechanisms

The code models three primary heat transfer mechanisms:

1. **Conduction** - Heat flow through solid materials
2. **Convection** - Heat transfer to/from fluids (air or liquid)
3. **Phase Change** - Evaporation/boiling for enhanced cooling

---

## 2. Material Properties and Temperature Dependence

### 2.1 ThermalLayer Class

```python
@dataclass
class ThermalLayer:
    """Thermal layer with material properties"""
    geometry: LayerInfo
    material: MaterialProperties
```

Each layer in the GPU package (die, TIM, IHS, etc.) has:
- **Thermal conductivity (k)** [W/m·K] - How well heat flows through the material
- **Density (ρ)** [kg/m³] - Mass per unit volume
- **Specific heat capacity (cp)** [J/kg·K] - Energy needed to raise temperature
- **Contact resistance** [K·W/m²] - Resistance at material interfaces

### 2.2 Temperature-Dependent Conductivity

```python
def thermal_resistance_at_temp(self, T: float) -> float:
    """Calculate temperature-dependent resistance"""
    k_T = self.material.k_temp_dependent(T)
    return self.geometry.thickness / (k_T * self.geometry.area)
```

**Physics**: Thermal conductivity varies with temperature. For semiconductors, it typically decreases with temperature due to increased phonon scattering. The relationship often follows:

k(T) = k₀ × (T₀/T)^n

Where n ≈ 1.2-1.5 for silicon.

### 2.3 Thermal Capacitance

```python
@property
def thermal_capacitance(self) -> float:
    """Calculate thermal capacitance of layer"""
    volume = self.geometry.thickness * self.geometry.area
    return self.material.rho * volume * self.material.cp
```

**Physics**: Thermal capacitance (C_th) determines transient response:
- C_th = ρ × V × cp
- Time constant: τ = R_th × C_th
- Important for understanding thermal lag and temperature oscillations

---

## 3. Thermal Resistance Calculations

### 3.1 Conduction Resistance

```python
@property
def thermal_resistance(self) -> float:
    """Calculate conduction resistance through layer"""
    return self.geometry.thickness / (self.material.k * self.geometry.area)
```

**Physics**: Based on Fourier's Law of heat conduction:
- q = -k × ∇T
- For 1D steady-state: R_cond = L / (k × A)
- Lower k or larger L increases resistance
- Larger area A reduces resistance

### 3.2 Convection Resistance

```python
@staticmethod
def convection(h: float, area: float) -> float:
    """Convection resistance"""
    return 1 / (h * area)
```

**Physics**: Based on Newton's Law of Cooling:
- q = h × A × (T_surface - T_fluid)
- R_conv = 1 / (h × A)
- Heat transfer coefficient h depends on:
  - Fluid properties (viscosity, thermal conductivity, density)
  - Flow conditions (laminar/turbulent, velocity)
  - Geometry (flat plate, fins, channels)

Typical h values:
- Natural air convection: 5-25 W/m²·K
- Forced air convection: 25-250 W/m²·K
- Liquid cooling: 1000-20,000 W/m²·K

### 3.3 Contact Resistance

```python
@staticmethod
def contact(material: MaterialProperties, area: float, 
            pressure: float = 1e6) -> float:
    """Contact resistance including material properties"""
    if material.contact_resistance:
        return material.contact_resistance / area
    return 1e-5 / area  # Simplified default
```

**Physics**: Real surfaces have microscopic roughness. Contact occurs only at asperities (high points), creating constriction resistance. Factors affecting contact resistance:
- Surface roughness and flatness
- Contact pressure (higher pressure → more contact → lower resistance)
- Interstitial material (air gaps vs thermal paste)
- Material hardness and thermal properties

### 3.4 Spreading Resistance

```python
@staticmethod
def spreading(k: float, source_area: float, sink_area: float, 
              thickness: float = 0) -> float:
    """Spreading resistance for area mismatch"""
```

**Physics**: When heat flows from a small source to a larger sink, streamlines must spread out, creating additional resistance. The Lee et al. correlation models this as:

- ε = r_source / r_sink (radius ratio)
- τ = thickness / r_sink (thickness parameter)
- ψ = tanh(τ) + ε/cosh(τ) (spreading factor)
- R_spread = ψ / (2π × k × r_source)

This is critical in GPU packages where the die is much smaller than the heat spreader.

---

## 4. GPU Power Dissipation Model

### 4.1 Dynamic Power

```python
def dynamic_power(self) -> float:
    """Calculate dynamic power dissipation"""
    return self.cores * self.alpha * self.C_eff * self.V_dd**2 * self.f_clock
```

**Physics**: CMOS dynamic power from charging/discharging capacitances:
- P_dynamic = α × C × V² × f
- α = activity factor (fraction of transistors switching)
- C = effective capacitance of all switching nodes
- V = supply voltage
- f = clock frequency

Power scales quadratically with voltage, making voltage reduction highly effective for power savings.

### 4.2 Leakage Power

```python
def leakage_power(self, T: float, I_leak_ref: float = 0.1, 
                  T_ref: float = 300, E_a: float = 0.5) -> float:
    """Temperature-dependent leakage power"""
    k_B = 8.617e-5  # Boltzmann constant [eV/K]
    I_leak = I_leak_ref * (T/T_ref)**2 * np.exp(-E_a/k_B * (1/T - 1/T_ref))
```

**Physics**: Subthreshold and gate leakage currents increase exponentially with temperature:
- Follows Arrhenius relationship: I ∝ exp(-E_a/k_B×T)
- E_a = activation energy (typically 0.4-0.6 eV)
- Creates positive feedback: Higher T → More leakage → More heat → Higher T
- Can lead to thermal runaway if cooling is insufficient

---

## 5. Advanced Cooling Methods

### 5.1 Evaporative/Boiling Cooling

```python
def nucleate_boiling_htc(self, q_flux: float, C_sf: float = 0.013) -> float:
    """Rohsenow correlation for nucleate boiling"""
```

**Physics**: Boiling provides exceptional heat transfer through:
1. **Nucleate boiling** - Bubbles form at surface nucleation sites
2. **Latent heat** - Phase change absorbs large amounts of energy
3. **Microconvection** - Bubble motion enhances fluid mixing

The Rohsenow correlation relates heat flux to wall superheat:
- Accounts for fluid properties (viscosity, surface tension, densities)
- C_sf = surface-fluid combination factor
- Prandtl number (Pr) effects included

### 5.2 Critical Heat Flux (CHF)

```python
def critical_heat_flux(self) -> float:
    """Zuber critical heat flux"""
    q_chf = 0.131 * self.fluid.h_fg * self.fluid.rho_v**0.5 * \
            (g * self.fluid.sigma * (self.fluid.rho_l - self.fluid.rho_v))**0.25
```

**Physics**: CHF represents the maximum heat flux before transition to film boiling:
- Beyond CHF, vapor blanket forms, dramatically reducing heat transfer
- Zuber's correlation based on hydrodynamic instability theory
- Depends on:
  - Latent heat of vaporization (h_fg)
  - Surface tension (σ)
  - Density difference between liquid and vapor
  - Gravity (buoyancy effects)

Operating near CHF is dangerous - sudden temperature excursions can damage components.

### 5.3 Vapor Chamber Model

```python
class VaporChamberModel:
    """Vapor chamber thermal model"""
    def effective_conductivity(self) -> float:
        k_eff = 10000  # Very high effective conductivity
```

**Physics**: Vapor chambers use two-phase heat transfer:
1. **Evaporation** at hot spot - liquid → vapor (absorbs heat)
2. **Vapor transport** - low resistance due to high vapor diffusivity
3. **Condensation** at cold regions - vapor → liquid (releases heat)
4. **Liquid return** via capillary action in wick structure

Effective thermal conductivity can be 10-100× higher than copper due to latent heat transport.

### 5.4 Capillary Limit

```python
def capillary_limit(self, fluid_key: str = 'water') -> float:
    """Maximum heat transfer limited by capillary pumping"""
    delta_P_cap = 2 * fluid.sigma / self.wick['pore_radius']
```

**Physics**: The wick must pump liquid against:
- Viscous pressure drop in liquid flow
- Vapor pressure drop
- Gravitational head (if present)

Maximum capillary pressure from Young-Laplace equation:
- ΔP_cap = 2σ/r_pore
- Smaller pores → higher pumping pressure
- Trade-off: smaller pores also increase flow resistance

---

## 6. Steady-State Thermal Solution

### 6.1 Thermal Network Assembly

```python
def calculate_steady_state(self, power: float, T_ambient: float,
                         cooling_type: CoolingType,
                         cooling_params: Dict) -> Dict[str, float]:
```

**Process**:
1. Sum all series resistances (layers + spreading + contact)
2. Add cooling resistance based on method
3. Apply power to find temperature rise: T_junction = T_ambient + Q × R_total
4. Calculate intermediate temperatures by stepping through network

### 6.2 Temperature Distribution

The temperature drops across each resistance according to:
- ΔT_i = Q × R_i
- Temperature is continuous, heat flux may jump at interfaces
- Highest temperature at junction, lowest at ambient/coolant

---

## 7. Nonlinear Thermal-Electrical Coupling

### 7.1 Iterative Solution

```python
def solve_nonlinear(self, power_model: PowerModel, T_ambient: float,
                   cooling_type: CoolingType, cooling_params: Dict,
                   max_iter: int = 50, tol: float = 0.1) -> Dict[str, float]:
```

**Physics**: The coupled problem arises from:
- Power depends on temperature (leakage)
- Temperature depends on power (thermal resistance)

Solution approach:
1. Guess initial T_junction
2. Calculate power at this temperature
3. Calculate new temperature from power
4. Iterate until convergence

This captures the positive feedback between temperature and leakage power.

---

## 8. Practical Design Considerations

### 8.1 Thermal Design Power (TDP)
- Must account for worst-case: maximum dynamic + leakage power
- Safety margin needed for:
  - Manufacturing variations
  - Aging effects
  - Ambient temperature variations

### 8.2 Hotspot Management
- Actual die has non-uniform power distribution
- Hotspots can be 10-20°C above average
- Spreading resistance becomes critical
- May need localized cooling enhancement

### 8.3 Reliability Implications
- Arrhenius relationship: 10°C increase ≈ 2× failure rate
- Thermal cycling causes mechanical stress
- Temperature gradients create differential expansion
- Junction temperature typically limited to 85-105°C

### 8.4 Transient Effects (Not modeled here)
- Thermal capacitance creates lag
- Important for:
  - Power throttling algorithms
  - Turbo boost duration
  - Thermal protection response time

---

## 9. Key Takeaways

1. **Thermal resistance network** provides intuitive analysis framework
2. **Temperature-dependent properties** create nonlinear behavior
3. **Contact and spreading resistances** often dominate in packages
4. **Leakage power** creates positive feedback requiring iterative solution
5. **Phase-change cooling** offers order-of-magnitude improvement but has limits
6. **System-level optimization** must balance multiple constraints

This thermal model captures the essential physics while remaining computationally tractable for design exploration and optimization.