# ClusterModel Architecture Implementation Guide

## Current Implementation
*Working on: Clustering implementation - `generate_clusters()` method*

---

## Implementation Log
### Configuration System [COMPLETED]
- ✅ Created `ClusterModelConfig` as frozen dataclass
- ✅ Added `PhysicalParams` dataclass for physics parameters  
- ✅ Implemented tuple-based ratios for k-spacing (numerator, denominator of 2π)
- ✅ Added `__post_init__` validation for L compatibility with ratios
- ✅ Fixed ham_lib parameter inconsistencies
- ✅ Added backend-specific handling in `solve_hamiltonian`

**Remaining Config Issues:**
- Consider adding convenience methods to get integer k-point separations from ratios
- May want to add factory methods for common configurations (half-filling, etc.)
- YAML/JSON loading can wait until core functionality works

---

## TODO List

### Phase 1: Core Configuration System [DONE - see log above]

### Phase 2: Clustering Strategies
- [ ] Define `ClusteringStrategy` abstract base class
- [ ] Implement `PeriodicClusteringStrategy` (extract from current ClusterExperiment)
- [ ] Implement `MismatchedClusteringStrategy` (extract from MismatchedQuick.recluster)
- [ ] Create unit tests for clustering strategies

### Phase 3: Hamiltonian Builder
- [ ] Create `HamiltonianBuilder` base class with fluent interface
- [ ] Extract term calculators (mu_tilde, hopping, interaction)
- [ ] Implement pattern classes (NearestNeighbor, Staggered, etc.)
- [ ] Create backend adapters (TenpyBackend, QuspinBackend)

### Phase 4: Solver Pipeline
- [ ] Define `SolverPipeline` abstract class with template method
- [ ] Implement `TenpySolver` (extract from current SpectrumSolver)
- [ ] Implement `QuspinSolver` (extract from QuSpinHamiltonian)
- [ ] Create `Spectrum` dataclass for standardized output

### Phase 5: Thermodynamics
- [ ] Create `ThermodynamicsCalculator` interface
- [ ] Implement ground state calculator
- [ ] Implement finite temperature calculator
- [ ] Add expectation value calculations

### Phase 6: Integration
- [ ] Implement main `ClusterModel` class
- [ ] Add `compute()` method
- [ ] Add `compute_batch()` for parameter sweeps
- [ ] Create factory classes for component creation

### Phase 7: Testing & Documentation
- [ ] Write comprehensive unit tests
- [ ] Create integration tests matching current functionality
- [ ] Add docstrings and type hints
- [ ] Create example notebooks

---

## Detailed Architecture Notes

### 🎯 Core Architectural Principles

**1. Single Responsibility + Dependency Inversion**
```python
# BAD (current approach):
class Hubbard1D:
    def init_terms(self, model_params):
        # Does EVERYTHING: validates, calculates mu_tilde, adds terms, handles V...
        
# GOOD (new approach):
class HamiltonianBuilder:
    def __init__(self, term_calculator: TermCalculator, term_adder: TermAdder):
        # Each dependency has ONE job
        
class MuTildeCalculator:
    def calculate(self, t: float, basis: Basis) -> float:
        # ONLY calculates mu_tilde
        
class HoppingTermAdder:
    def add_terms(self, builder: MPOBuilder, params: HoppingParams):
        # ONLY adds hopping terms
```

**2. Configuration as Code**
```python
# Everything flows from configuration
@dataclass(frozen=True)  # Immutable configs prevent bugs
class ClusterModelConfig:
    """Single source of truth for ALL parameters"""
    clustering: ClusteringConfig
    physics: PhysicsConfig  
    solver: SolverConfig
    thermodynamics: ThermodynamicsConfig
    
    def validate(self) -> None:
        """Fail fast with clear messages"""
        if self.clustering.size > self.physics.system_size:
            raise ValueError(f"Cluster size {self.clustering.size} exceeds system")
```

### 📐 Layer Architecture

```
┌─────────────────────────────────────┐
│         ClusterModel (API)          │ <- User interacts here
├─────────────────────────────────────┤
│        Orchestration Layer          │ <- Coordinates components
├─────────────────────────────────────┤
│     Domain Services (Strategies)    │ <- Business logic
├─────────────────────────────────────┤
│      Infrastructure (Solvers)       │ <- External dependencies
└─────────────────────────────────────┘
```

### 🔧 Detailed Implementation Patterns

**1. Strategy Pattern for Clustering**
```python
from abc import ABC, abstractmethod

class ClusteringStrategy(ABC):
    @abstractmethod
    def generate_clusters(self, system_size: int) -> np.ndarray:
        pass
        
class PeriodicClusteringStrategy(ClusteringStrategy):
    def __init__(self, cluster_size: int, k_spacing: float):
        self.cluster_size = cluster_size
        self.k_spacing = k_spacing
        
    def generate_clusters(self, system_size: int) -> np.ndarray:
        # Core logic from current ClusterExperiment
        
class MismatchedClusteringStrategy(ClusteringStrategy):
    """For V-coupling that doesn't match interaction clustering"""
    def __init__(self, int_clusters: ClusteringStrategy, v_period: float):
        self.int_clusters = int_clusters
        self.v_period = v_period
        
    def generate_clusters(self, system_size: int) -> np.ndarray:
        # Logic from MismatchedQuick.recluster()
```

**Principle**: Open/Closed - Add new clustering without modifying existing code

**2. Builder Pattern for Hamiltonians**
```python
class HamiltonianBuilder:
    """Fluent interface for constructing Hamiltonians"""
    
    def __init__(self, backend: HamiltonianBackend):
        self.backend = backend
        self.terms = []
        
    def add_hopping(self, t: float, pattern: HoppingPattern) -> 'HamiltonianBuilder':
        """Add hopping terms based on pattern"""
        self.terms.append(HoppingTerm(t, pattern))
        return self
        
    def add_interaction(self, U: float, sites: List[int]) -> 'HamiltonianBuilder':
        self.terms.append(InteractionTerm(U, sites))
        return self
        
    def add_potential(self, V: float, pattern: PotentialPattern) -> 'HamiltonianBuilder':
        self.terms.append(PotentialTerm(V, pattern))
        return self
        
    def build(self) -> Hamiltonian:
        return self.backend.construct(self.terms)

# Usage:
ham = (HamiltonianBuilder(TenpyBackend())
       .add_hopping(t=1.0, pattern=NearestNeighbor())
       .add_interaction(U=5.0, sites=all_sites)
       .add_potential(V=2.0, pattern=Staggered())
       .build())
```

**Principle**: Separation of Concerns - Building logic separate from backend implementation

**3. Template Method for Solver Pipeline**
```python
class SolverPipeline(ABC):
    """Template for all solving procedures"""
    
    def solve(self, hamiltonian: Hamiltonian) -> Spectrum:
        # Template method defining algorithm structure
        prepared_ham = self._prepare_hamiltonian(hamiltonian)
        raw_solution = self._execute_solver(prepared_ham)
        return self._post_process(raw_solution)
        
    @abstractmethod
    def _prepare_hamiltonian(self, ham: Hamiltonian) -> Any:
        """Backend-specific preparation"""
        
    @abstractmethod  
    def _execute_solver(self, ham: Any) -> Any:
        """Actual diagonalization"""
        
    def _post_process(self, solution: Any) -> Spectrum:
        """Convert to standard format"""
        return Spectrum(
            energies=self._extract_energies(solution),
            states=self._extract_states(solution),
            occupations=self._calculate_occupations(solution)
        )

class TenpySolver(SolverPipeline):
    def _prepare_hamiltonian(self, ham: Hamiltonian) -> MPO:
        # Current logic from SpectrumSolver
        
class QuspinSolver(SolverPipeline):
    def _prepare_hamiltonian(self, ham: Hamiltonian) -> csr_matrix:
        # Current logic from QuSpinHamiltonian
```

**Principle**: Don't Repeat Yourself - Common algorithm, varying implementations

**4. Factory Pattern for Component Creation**
```python
class ComponentFactory:
    """Creates configured components based on config"""
    
    @staticmethod
    def create_clustering(config: ClusteringConfig) -> ClusteringStrategy:
        if config.type == "periodic":
            return PeriodicClusteringStrategy(config.size, config.spacing)
        elif config.type == "mismatched":
            return MismatchedClusteringStrategy(...)
        else:
            raise ValueError(f"Unknown clustering: {config.type}")
            
    @staticmethod
    def create_solver(config: SolverConfig) -> SolverPipeline:
        solvers = {
            "tenpy": TenpySolver,
            "quspin": QuspinSolver,
            "exact": ExactDiagSolver
        }
        return solvers[config.backend](config.params)
```

**Principle**: Dependency Injection - Components don't create their dependencies

### 🏗️ The Complete ClusterModel

```python
class ClusterModel:
    """The unified interface for all cluster calculations"""
    
    def __init__(self, config: ClusterModelConfig):
        self.config = config
        self._validate_configuration()
        
        # Dependency injection via factory
        self.clustering = ComponentFactory.create_clustering(config.clustering)
        self.ham_builder = ComponentFactory.create_ham_builder(config.physics)
        self.solver = ComponentFactory.create_solver(config.solver)
        self.thermo_calc = ComponentFactory.create_thermo(config.thermodynamics)
        
    def compute(self) -> Results:
        """One method to rule them all"""
        # Generate cluster structure
        clusters = self.clustering.generate_clusters(self.config.physics.system_size)
        
        # Build Hamiltonians for each cluster
        hamiltonians = [
            self.ham_builder.build_for_cluster(cluster, self.config.physics)
            for cluster in clusters
        ]
        
        # Solve for spectra
        spectra = [self.solver.solve(ham) for ham in hamiltonians]
        
        # Calculate thermodynamics
        return self.thermo_calc.calculate(spectra, self.config.thermodynamics.temperature)
        
    def compute_batch(self, configs: List[ClusterModelConfig]) -> List[Results]:
        """Efficient batch processing"""
        return [ClusterModel(cfg).compute() for cfg in configs]
```

### 🔑 Key Implementation Details

**1. Reuse Current Logic**
- Extract `mu_tilde` calculation into `TermCalculators.calculate_chemical_potential()`
- Move clustering algorithm from `ClusterExperiment` into `PeriodicClusteringStrategy`
- Extract V-coupling logic from `QuickHubbard1D` into `StaggeredPotentialPattern`

**2. Error Handling**
```python
class ClusteringError(Exception):
    """Raised when clusters can't tile the Brillouin zone"""
    
class ConvergenceError(Exception):
    """Raised when solver doesn't converge"""
    
# Use specific exceptions for specific problems
```

**3. Testing Strategy**
```python
# Each component is independently testable
def test_periodic_clustering():
    strategy = PeriodicClusteringStrategy(size=2, spacing=np.pi)
    clusters = strategy.generate_clusters(system_size=10)
    assert_valid_tiling(clusters)
    
# Integration tests use the full model
def test_full_calculation():
    config = ClusterModelConfig.from_yaml("test_config.yaml")
    model = ClusterModel(config)
    results = model.compute()
    assert_energy_conservation(results)
```

**4. Progressive Enhancement**
Start with core functionality, add features incrementally:
1. Basic periodic clustering → Add mismatched clustering
2. Simple Hubbard model → Add extended interactions
3. Ground state only → Add finite temperature
4. Single calculation → Add parameter sweeps

This architecture ensures that your `ClusterModel` is:
- **Extensible**: Add new strategies without breaking existing code
- **Testable**: Every component can be tested in isolation
- **Maintainable**: Clear separation of concerns
- **Performant**: Batch processing and caching where needed
- **Understandable**: One-line API with clear configuration