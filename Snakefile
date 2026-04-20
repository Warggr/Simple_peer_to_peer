rule make_data:
    output: "data/saved_test_result_{seed}.pkl"
    shell: """python main.py {wildcards.seed} {output}"""

rule plot_data:
    input: "data/saved_test_result_0.pkl"
    output: [f"Figures/{what}.{ext}" for what in ('Constr_viol', 'Distance_tvar', 'Residual') for ext in ('png', 'pdf')]
    shell: """python plot.py {input}"""
