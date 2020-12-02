braket.pennylane_plugin.BraketLocalQubitDevice
==============================================

.. currentmodule:: braket.pennylane_plugin

.. autoclass:: BraketLocalQubitDevice
   :show-inheritance:

   .. raw:: html

      <a class="attr-details-header collapse-header" data-toggle="collapse" href="#attrDetails" aria-expanded="false" aria-controls="attrDetails">
         <h2 style="font-size: 24px;">
            <i class="fas fa-angle-down rotate" style="float: right;"></i> Attributes
         </h2>
      </a>
      <div class="collapse" id="attrDetails">

   .. autosummary::
      :nosignatures:

      ~BraketLocalQubitDevice.author
      ~BraketLocalQubitDevice.cache
      ~BraketLocalQubitDevice.circuit
      ~BraketLocalQubitDevice.circuit_hash
      ~BraketLocalQubitDevice.name
      ~BraketLocalQubitDevice.num_executions
      ~BraketLocalQubitDevice.obs_queue
      ~BraketLocalQubitDevice.observables
      ~BraketLocalQubitDevice.op_queue
      ~BraketLocalQubitDevice.operations
      ~BraketLocalQubitDevice.parameters
      ~BraketLocalQubitDevice.pennylane_requires
      ~BraketLocalQubitDevice.short_name
      ~BraketLocalQubitDevice.shots
      ~BraketLocalQubitDevice.state
      ~BraketLocalQubitDevice.task
      ~BraketLocalQubitDevice.version
      ~BraketLocalQubitDevice.wire_map
      ~BraketLocalQubitDevice.wires

   .. autoattribute:: author
   .. autoattribute:: cache
   .. autoattribute:: circuit
   .. autoattribute:: circuit_hash
   .. autoattribute:: name
   .. autoattribute:: num_executions
   .. autoattribute:: obs_queue
   .. autoattribute:: observables
   .. autoattribute:: op_queue
   .. autoattribute:: operations
   .. autoattribute:: parameters
   .. autoattribute:: pennylane_requires
   .. autoattribute:: short_name
   .. autoattribute:: shots
   .. autoattribute:: state
   .. autoattribute:: task
   .. autoattribute:: version
   .. autoattribute:: wire_map
   .. autoattribute:: wires

   .. raw:: html

      </div>

   .. raw:: html

      <a class="meth-details-header collapse-header" data-toggle="collapse" href="#methDetails" aria-expanded="false" aria-controls="methDetails">
         <h2 style="font-size: 24px;">
            <i class="fas fa-angle-down rotate" style="float: right;"></i> Methods
         </h2>
      </a>
      <div class="collapse" id="methDetails">

   .. autosummary::

      ~BraketLocalQubitDevice.access_state
      ~BraketLocalQubitDevice.active_wires
      ~BraketLocalQubitDevice.analytic_probability
      ~BraketLocalQubitDevice.apply
      ~BraketLocalQubitDevice.batch_execute
      ~BraketLocalQubitDevice.capabilities
      ~BraketLocalQubitDevice.check_validity
      ~BraketLocalQubitDevice.define_wire_map
      ~BraketLocalQubitDevice.density_matrix
      ~BraketLocalQubitDevice.estimate_probability
      ~BraketLocalQubitDevice.execute
      ~BraketLocalQubitDevice.execution_context
      ~BraketLocalQubitDevice.expval
      ~BraketLocalQubitDevice.generate_basis_states
      ~BraketLocalQubitDevice.generate_samples
      ~BraketLocalQubitDevice.map_wires
      ~BraketLocalQubitDevice.marginal_prob
      ~BraketLocalQubitDevice.post_apply
      ~BraketLocalQubitDevice.post_measure
      ~BraketLocalQubitDevice.pre_apply
      ~BraketLocalQubitDevice.pre_measure
      ~BraketLocalQubitDevice.probability
      ~BraketLocalQubitDevice.reset
      ~BraketLocalQubitDevice.sample
      ~BraketLocalQubitDevice.sample_basis_states
      ~BraketLocalQubitDevice.states_to_binary
      ~BraketLocalQubitDevice.statistics
      ~BraketLocalQubitDevice.supports_observable
      ~BraketLocalQubitDevice.supports_operation
      ~BraketLocalQubitDevice.var

   .. automethod:: access_state
   .. automethod:: active_wires
   .. automethod:: analytic_probability
   .. automethod:: apply
   .. automethod:: batch_execute
   .. automethod:: capabilities
   .. automethod:: check_validity
   .. automethod:: define_wire_map
   .. automethod:: density_matrix
   .. automethod:: estimate_probability
   .. automethod:: execute
   .. automethod:: execution_context
   .. automethod:: expval
   .. automethod:: generate_basis_states
   .. automethod:: generate_samples
   .. automethod:: map_wires
   .. automethod:: marginal_prob
   .. automethod:: post_apply
   .. automethod:: post_measure
   .. automethod:: pre_apply
   .. automethod:: pre_measure
   .. automethod:: probability
   .. automethod:: reset
   .. automethod:: sample
   .. automethod:: sample_basis_states
   .. automethod:: states_to_binary
   .. automethod:: statistics
   .. automethod:: supports_observable
   .. automethod:: supports_operation
   .. automethod:: var

   .. raw:: html

      </div>

   .. raw:: html

      <script type="text/javascript">
         $(".collapse-header").click(function () {
             $(this).children('h2').eq(0).children('i').eq(0).toggleClass("up");
         })
      </script>
