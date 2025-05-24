from fastapi import HTTPException, UploadFile

from app.services.acceleration_speed import generate_window_feedback, pipeline_compare_users, feedback_window_metrics_acel_vel
from app.services.align_movements import alinear_usuarios_procrustes, sync_two_users_dict, convertir_numpy_a_listas
from app.services.calculate_errors import crear_dataframes_movimientos, generar_diccionario_diferencias, generar_diccionario_diferencias_angulares, rangos_movimientos_con_signos
from app.services.error_clustering import agregar_clusters_por_error, agregar_clusters_por_error_angular_v2, agrupar_por_ventanas_ms, etiquetar_por_ventanas_fijas
from app.services.feedback_builder import transform_dict_pos_ang_to_text, transform_dict_vel_acel_to_text, get_feedback_from_type_data_external_IA
from app.services.joint_centers_angles import convertir_a_dict_articulaciones, create_dict_with_data
from .processing_files import add_data_to_dict, load_excel_file, create_tuple_names_list, creating_marker_dict


async def execute_comparison(base_file: UploadFile, compare_file: UploadFile, joint_id: str):
    try:
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
        resultados_ventanas_angles = etiquetar_por_ventanas_fijas(dict_labels_clustering_joint_individual_angular)
        acel_vel_dict = pipeline_compare_users(user_one_to_model, user_two_to_model, window_ms=3)
        feedback_acel_vel_dict = feedback_window_metrics_acel_vel(acel_vel_dict)

        text_position_to_ia = transform_dict_pos_ang_to_text(resultados_ventanas_joints)
        text_angular_to_ia = transform_dict_pos_ang_to_text(resultados_ventanas_angles)
        text_vel_ac_to_ia = transform_dict_vel_acel_to_text(feedback_acel_vel_dict)

        feedback_position_external = await get_feedback_from_type_data_external_IA(text_position_to_ia, "position")
        feedback_angular_external = await get_feedback_from_type_data_external_IA(text_angular_to_ia, "angular")
        feedback_acel_vel_external = await get_feedback_from_type_data_external_IA(text_vel_ac_to_ia, "acel_vel")

        user_one_full_to_response = convertir_numpy_a_listas(user_one_to_model)
        user_two_full_to_response = convertir_numpy_a_listas(user_two_to_model)

        response = {
            "feedback_position": feedback_position_external,
            "feedback_angular": feedback_angular_external,
            "feedback_acel_vel": feedback_acel_vel_external,
            "values_position": resultados_ventanas_joints,
            "values_angular": resultados_ventanas_angles,
            "values_acel_vel": feedback_acel_vel_dict,
            "data_full_user_one": user_one_full_to_response,
            "data_full_user_two": user_two_full_to_response,
        }

        return response

    except Exception as e:  
        print(e)
        raise HTTPException(status_code=500, detail=f"Error al ejecutar servicio principal - FastAPI.\n {e}")