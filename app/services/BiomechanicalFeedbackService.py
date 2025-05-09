from fastapi import UploadFile

from app.services.AccelerationSpeed import generate_window_feedback, pipeline_compare_users
from app.services.AlignMovements import alinear_usuarios_procrustes, sync_two_users_dict
from app.services.CalculateErrors import crear_dataframes_movimientos, generar_diccionario_diferencias, generar_diccionario_diferencias_angulares, rangos_movimientos_con_signos
from app.services.ErrorClustering import agregar_clusters_por_error, agregar_clusters_por_error_angular_v2, agrupar_por_ventanas_ms, etiquetar_por_ventanas_fijas
from app.services.JointCentersAngles import convertir_a_dict_articulaciones, create_dict_with_data
from .ProcessingFiles import add_data_to_dict, load_excel_file, create_tuple_names_list, creating_marker_dict  # Replace 'some_module' with the actual module name where the function is defined.


async def execute_comparison(base_file: UploadFile, compare_file: UploadFile, joint_id: str):
    # Placeholder for the actual comparison logic
    print("Executing biomechanical feedback comparison...")
    user_one_marker_df = await load_excel_file(base_file)
    user_two_marker_df = await load_excel_file(compare_file)

    marker_ndates_tuple_list_one = create_tuple_names_list(user_one_marker_df)
    marker_ndates_tuple_list_two = create_tuple_names_list(user_two_marker_df)

    marker_multiindex_one = creating_marker_dict(user_one_marker_df, marker_ndates_tuple_list_one)
    marker_multiindex_two = creating_marker_dict(user_two_marker_df, marker_ndates_tuple_list_two)
    
    marker_one_full = add_data_to_dict(user_one_marker_df, marker_multiindex_one,marker_ndates_tuple_list_one, 1)
    marker_two_full = add_data_to_dict(user_two_marker_df, marker_multiindex_two, marker_ndates_tuple_list_two,1)
    
    art_center_one = convertir_a_dict_articulaciones(marker_one_full, create_dict_with_data(marker_one_full), step=1)
    art_center_two = convertir_a_dict_articulaciones(marker_two_full, create_dict_with_data(marker_two_full), step=1)
    
    user_one_to_model, user_two_to_model = sync_two_users_dict( art_center_one, alinear_usuarios_procrustes(art_center_one, art_center_two), joint_id)
    
    movements_two_dict = crear_dataframes_movimientos(user_two_to_model, rangos_movimientos_con_signos, False)

    diferencias_posicion_dict = generar_diccionario_diferencias(user_two_to_model, user_one_to_model)
    diferencias_angles_dict = generar_diccionario_diferencias_angulares(movements_two_dict, user_one_to_model)


    dict_labels_clustering_joint_individual = agregar_clusters_por_error(diferencias_posicion_dict)
    resultados_ventanas_joints = agrupar_por_ventanas_ms(dict_labels_clustering_joint_individual, duracion_ventana_ms=1)
    dict_labels_clustering_joint_individual_angular = agregar_clusters_por_error_angular_v2(diferencias_angles_dict)
    etiquetar_por_ventanas_fijas(dict_labels_clustering_joint_individual_angular)
    
    acel_vel_dict = pipeline_compare_users(user_one_to_model, user_two_to_model, window_ms=1)

    for joint in acel_vel_dict:
    
        print(f"\nArticulación: {joint}")
        df_feedback = generate_window_feedback(acel_vel_dict[joint])
        for idx, texto in df_feedback['feedback'].items():
            print(f"{idx}: {texto}")
    
        print("\n\n\n")