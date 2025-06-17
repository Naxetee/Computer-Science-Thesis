from PyARXaaS.qids_selection import qids_selection
import os
import argparse


def run_qids_selection(k:int, l:int, t:float, q, debug_comments:bool) -> None:
    """
    Función para ejecutar la selección de QIDs con los parámetros especificados.
    Args:
        k (int): Valor de k para K-Anonymity.
        l (int): Valor de l para l-Diversity.
        t (float): Valor de t para t-Closeness.
        q (int): Número máximo de QIDs a seleccionar.
        debug_comments (bool): Si es True, activa el modo de depuración.
    """
    qids_selection(
                sensitive_attribute='income',
                mandatory_qids= ['sex', 'age'],
                insensitive_attributes = ['fnlwgt'],
                anonymity_models= {
                    'k': k,
                    'l': l,
                    't': t
                },
                dataset_path='./data/adults/original/adults_rounded.csv',
                output_folder='./data/adults/anonymized/',
                max_num_of_qids=q,
                hierarchy_folder='./data/adults/hierarchies/',
                output_file=f"./data/adults/results/anonymization/{args.q}_qids/k_{k}-l_{l}-t_{t}-results.csv",
                ARXaaS_port=8080,
                debug=debug_comments,
                export=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run QIDs selection with various anonymity parameters.")
    parser.add_argument('--output_folder', type=str, default='./data/adults/results/anonymization/', help='Output folder for anonymized data')
    parser.add_argument('--k', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32, 64], help='List of k values for K-Anonymity')
    parser.add_argument('--l', type=int, nargs='+', default=[1, 2], help='List of l values for l-Diversity')
    parser.add_argument('--t', type=float, nargs='+', default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], help='List of t values for t-Closeness')
    parser.add_argument('--q', type=int, default=4, help='Maximum number of QIDs to select')
    parser.add_argument('--debug_comments', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    if args.debug_comments:
        print("Modo depuración activado")
        input("Presiona Enter para continuar...")

    # Verificar si el directorio de salida existe, si no, crearlo
    output_folder = args.output_folder + f"{args.q}_qids/"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        if args.debug_comments: print(f"Directorio {output_folder} creado.")

    for k in args.k:
        for l in args.l:
            for t in args.t:
                if args.debug_comments: print(f"Running with k={k}, l={l}, t={t}")
                output_file = f"{output_folder}k_{k}-l_{l}-t_{t}-results.csv"
                if os.path.exists(output_file):
                    if args.debug_comments: print(f"File {output_file} already exists. Skipping.")
                    continue
                run_qids_selection(k, l, t, args.q, args.debug_comments)
            