import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import keras

data = pd.read_csv('/app/Data/fifa23_players.csv')
# data = pd.read_csv('/Users/home/DataScience/coursePr/Data/fifa23_players.csv')
data = data.astype(int, errors='ignore')
urls = [idx for idx, col in enumerate(data.columns) if 'url' in col]
urls = [data.columns[idx] for idx in urls]
data = data.drop(columns= urls)
data_duplicated = pd.read_csv('/app/Data/duplicated_data.csv')
# data_duplicated = pd.read_csv('/Users/home/DataScience/coursePr/Data/duplicated_data.csv')

encoder_body = joblib.load('/app/sup_files/Encoder_body.sav')
encoder_work = joblib.load('/app/sup_files/Encoder_work.sav')
encoder_foot = joblib.load('/app/sup_files/Encoder_foot.sav')

# encoder_body = joblib.load('/Users/home/DataScience/coursePr/sup_files/Encoder_body.sav')
# encoder_work = joblib.load('/Users/home/DataScience/coursePr/sup_files/Encoder_work.sav')
# encoder_foot = joblib.load('/Users/home/DataScience/coursePr/sup_files/Encoder_foot.sav')


scaler = joblib.load('/app/sup_files/Scaler.sav')
# scaler = joblib.load('/Users/home/DataScience/coursePr/sup_files/Scaler.sav')

def top_100_overall(df):
    top_100_overall = df.nlargest(100, 'overall')
    return top_100_overall

def predict(x, pipeline):
    result = pipeline.predict(x)
    return result

def evaluate(x, pipeline):
    loss, predict = pipeline.evaluate(x)
    return loss, predict

def get_proba(x,  pipeline):
    result = pipeline.predict_proba(x)
    return result

def top_100_wage(df):
    top_100_wage = df.nlargest(100, 'wage_eur')
    return top_100_wage

def top_30_gk(df):
    top_30_gk = df[df['player_positions'] == 'GK'].nlargest(30, 'overall')
    return top_30_gk

def top_30_clubs_by_player_overall(df):
    top_30_clubs_by_player_overall = df.groupby('club_name')['overall'].max().nlargest(30)
    return top_30_clubs_by_player_overall

def top_30_clubs_sum_players_overall(df):
    top_30_clubs_sum_players_overall = df.groupby('club_name')['overall'].mean().nlargest(30)
    return top_30_clubs_sum_players_overall

def top_30_clubs_avg_pace(df):
    top_30_clubs_avg_pace = df.groupby('club_name')['pace'].mean().nlargest(30)
    return top_30_clubs_avg_pace

def top_leagues_dribbling(df):
    top_leagues_dribbling = df.groupby('league_name')['dribbling'].mean().sort_values(ascending=False)
    return top_leagues_dribbling

def top_teams_by_top_players(df):
    # Reset the index to make 'club_name' a regular column
    df_reset = df.reset_index()

    # Filter players by positions and group by 'club_name'
    goalkeepers = df_reset[df_reset['player_positions'] == 'GK'].groupby('club_name').apply(lambda x: x.nlargest(1, 'overall')).reset_index(drop=True)
    defenders = df_reset[df_reset['player_positions'].str.contains('RWB|CDM|LWB|LB|CB|RB')].groupby('club_name').apply(lambda x: x.nlargest(4, 'overall')).reset_index(drop=True)
    midfielders = df_reset[df_reset['player_positions'].str.contains('CAM|LW|RW|RM|CM|LM')].groupby('club_name').apply(lambda x: x.nlargest(4, 'overall')).reset_index(drop=True)
    strikers = df_reset[df_reset['player_positions'].str.contains('ST|CF')].groupby('club_name').apply(lambda x: x.nlargest(2, 'overall')).reset_index(drop=True)

    # Concatenate filtered data
    top_players = pd.concat([goalkeepers, defenders, midfielders, strikers])

    # Group by 'club_name' and compute the overall rating sum
    top_teams = top_players.groupby('club_name')['overall'].sum().nlargest(30)
    return top_teams

def distribution_by_role_age(df):
    df['player_role'] = df['player_positions'].apply(get_player_role)
    positions = ['CF', 'ST', 'RW', 'LW', 'CAM', 'RM', 'CM', 'LM', 'RWB', 'CDM', 'LWB', 'LB', 'CB', 'RB', 'GK']
    filtered_data = df[df['player_role'] != 'Unknown']


    plt.figure(figsize=(12, 6))
    sns.boxplot(x='player_role', y='age', data=filtered_data)
    plt.xlabel('Роль гравця')
    plt.ylabel('Вік')
    plt.title('Розподіл віку гравців за ролями')
    plt.xticks(rotation=45)
    plt.show()

def distribution_by_age_pos(df):
    positions = ['CF', 'ST', 'RW', 'LW', 'CAM', 'RM', 'CM', 'LM', 'RWB', 'CDM', 'LWB', 'LB', 'CB', 'RB', 'GK']
    data = df[df.player_positions.isin(positions)]
    plt.figure(figsize=(12,6))
    sns.boxenplot(data = data, x = 'player_positions', y = 'age')
    plt.xlabel('Позиція')
    plt.ylabel('Вік')
    plt.title('Розподіл віку гравців за позиціями')
    plt.xticks(rotation=45)
    plt.show()

def distribution_by_overall_pos(df):
    positions = ['CF', 'ST', 'RW', 'LW', 'CAM', 'RM', 'CM', 'LM', 'RWB', 'CDM', 'LWB', 'LB', 'CB', 'RB', 'GK']
    data = df[df.player_positions.isin(positions)]

    plt.figure(figsize=(12,6))
    sns.boxenplot(data = data, x = 'player_positions', y = 'overall')
    plt.xlabel('Позиція')
    plt.ylabel('Рейтинг')
    plt.title('Розподіл рейтингу гравців за позиціями')
    plt.xticks(rotation=45)
    plt.show()

def dis_by_nationalities(df):
    nationalities = [nation for nation, count in df['nationality_name'].value_counts().items() if count > 210]
    len(nationalities)
    data = df[df['nationality_name'].isin(nationalities)]
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=data, x='nationality_name', y='overall')
    plt.xlabel('Національність')
    plt.ylabel('Рейтинг')
    plt.title('Розподіл рейтингу гравців за національністю')
    plt.xticks(rotation=90)
    plt.show()

def penalty_score(df):
    
    data = df
    data['Penaltyist_Score'] = (0.4 * df['mentality_penalties'] + 0.4 * df['shooting'] + 0.1 * df['power_shot_power'] + 0.1 * df['mentality_vision']) / 4
    top_clubs = df.groupby('club_name')['Penaltyist_Score'].mean().nlargest(10)
    return top_clubs

def penalty_score_gk(df):
    data = df
    data['GK_penalty_score'] = (0.3 * data.goalkeeping_diving + 0.1 * data.goalkeeping_handling + 0.3 * data.goalkeeping_reflexes + 0.2 * data.goalkeeping_speed + 0.1 * data.mentality_penalties) / 5
    top_clubs = df.groupby('club_name')['GK_penalty_score'].mean().nlargest(10)
    return top_clubs

def get_player_role(positions):
    if 'ST' in positions or 'CF' in positions:
        return 'Attack Player'
    elif 'CAM' in positions or 'LW' in positions or 'RW' in positions or 'RM' in positions or 'CM' in positions or 'LM' in positions:
        return 'Midfielder'
    elif 'RWB' in positions or 'CDM' in positions or 'LWB' in positions or 'LB' in positions or 'CB' in positions or 'RB' in positions:
        return 'Defender'
    elif 'GK' in positions:
        return 'Goalkeeper'
    else:
        return 'Unknown'


def main():
    st.title('Fifa players positions prediction')
    
    # model_choice = ['neural(one hot encoded)', 'Random Forest Classifier(duplicates)', 'Random Forest Classifier(One hot encoded)']
    # choice = st.selectbox('Choose the model: ',  model_choice)

    # if choice == 'neural(one hot encoded)':
    #     pipeline = keras.models.load_model('/Users/home/DataScience/coursePr/models')
    
    # # if choice == 'Random Forest Classifier(duplicates)':
    # #     pipeline = joblib.load('/models/RandomForest_for_duplicated_Taget.sav')

    # # if choice == 'Random Forest Classifier(One hot encoded)':
    # #     pipeline = joblib.load('/models/RandomForest_for_One_Hot_Encoded_Taget.sav')

    if st.checkbox('Show DataFrame'):
        data

    st.subheader('Specify features:')

    left_column, right_column = st.columns(2)

    with left_column:
        preferred_foot = st.radio(
            'preferred_foot: ',
            np.unique(data['preferred_foot'])
        )
        work_rate = st.radio(
            'work_rate: ',
            np.unique(data['work_rate'])
        )
        body_type = st.radio(
            'body_type: ',
            np.unique(data['body_type'])
        )
    overall = st.slider('Overall : ', int(data.overall.min()), int(data['overall'].max()))
    age = st.slider('Age : ', int(min(data.age)), int(max(data['age'])))
    club_jersey_number = st.slider('Club number : ',int(min(data.club_jersey_number)), int(max(data['club_jersey_number'])))
    height_cm = st.slider('Height cm : ', 155.0, float(max(data['height_cm'])))    
    weight_kg = st.slider('Weight kg : ', int(min(data.weight_kg)), int(max(data['weight_kg'])))
    weak_foot = st.slider('Weak Foot : ', min(data['weak_foot']), max(data['weak_foot']))    
    value_eur = st.slider('Value : ', int(min(data.value_eur)), int(max(data['value_eur'])))
    wage_eur = st.slider('Wage : ', min(data.wage_eur), max(data['wage_eur']))
    skill_moves = st.slider('Skill moves : ', min(data.skill_moves), max(data['skill_moves']))
    pace = st.slider('Pace : ', 0, int(max(data['pace'])))
    shooting = st.slider('Shooting : ', 0, int(max(data['shooting'])))
    passing = st.slider('Passing : ', 0, int(max(data['passing'])))
    dribbling = st.slider('Dribbling : ', 0, int(max(data['dribbling'])))
    defending = st.slider('Defending : ', 0, int(max(data['defending'])))
    physic = st.slider('Physic : ', 0, int(max(data['physic'])))
    attacking_crossing = st.slider('Crossing : ', min(data.attacking_crossing), max(data['attacking_crossing']))
    attacking_finishing = st.slider('Finishing : ', min(data.attacking_finishing), max(data['attacking_finishing']))
    attacking_heading_accuracy = st.slider('Heading Accuracy : ', min(data.attacking_heading_accuracy), max(data['attacking_heading_accuracy']))
    attacking_short_passing = st.slider('Short passing : ', min(data.attacking_short_passing), max(data['attacking_short_passing']))
    attacking_volleys = st.slider('Attacking volleys : ', min(data.attacking_volleys), max(data['attacking_volleys']))
    skill_dribbling = st.slider('Skill dribbling : ', min(data.skill_dribbling), max(data['skill_dribbling']))
    skill_curve = st.slider('Curve : ', min(data.skill_curve), max(data['skill_curve']))
    skill_fk_accuracy = st.slider('FK accuracy : ', min(data.skill_fk_accuracy), max(data['skill_fk_accuracy']))
    skill_long_passing = st.slider('Long paassing : ', min(data.skill_long_passing), max(data['skill_long_passing']))
    skill_ball_control = st.slider('Ball control : ', min(data.skill_ball_control), max(data['skill_ball_control']))
    movement_acceleration = st.slider('Acceleration : ', min(data.movement_acceleration), max(data['movement_acceleration']))
    movement_sprint_speed = st.slider('Sprint speed : ', min(data.movement_sprint_speed), max(data['movement_sprint_speed']))
    movement_agility = st.slider('Agility : ', min(data.movement_agility), max(data['movement_agility']))
    movement_reactions = st.slider('Reactions : ', min(data.movement_reactions), max(data['movement_reactions']))
    movement_balance = st.slider('Balance : ', min(data.movement_balance), max(data['movement_balance']))
    power_shot_power = st.slider('Shot power : ', min(data.power_shot_power), max(data['power_shot_power']))
    power_jumping = st.slider('Jumping : ', min(data.power_jumping), max(data['power_jumping']))
    power_stamina = st.slider('Stamina : ', min(data.power_stamina), max(data['power_stamina']))
    power_strength = st.slider('Strength : ', min(data.power_strength), max(data['power_strength']))
    power_long_shots = st.slider('Long shots : ', min(data.power_long_shots), max(data['power_long_shots']))
    mentality_aggression = st.slider('Aggression : ', min(data.mentality_aggression), max(data['mentality_aggression']))
    mentality_interceptions = st.slider('Interceptions : ', min(data.mentality_interceptions), max(data['mentality_interceptions']))
    mentality_positioning = st.slider('Positioning : ', min(data.mentality_positioning), max(data['mentality_positioning']))
    mentality_vision = st.slider('Vision : ', min(data.mentality_vision), max(data['mentality_vision']))
    mentality_penalties = st.slider('Penalties : ', min(data.mentality_penalties), max(data['mentality_penalties']))
    mentality_composure = st.slider('Composure : ', min(data.mentality_composure), max(data['mentality_composure']))
    defending_marking_awareness = st.slider('Marking awarness : ', min(data.defending_marking_awareness), max(data['defending_marking_awareness']))
    defending_standing_tackle = st.slider('Standing tackle : ', min(data.defending_standing_tackle), max(data['defending_standing_tackle']))
    defending_sliding_tackle = st.slider('Sliding tackle : ', min(data.defending_sliding_tackle), max(data['defending_sliding_tackle']))
    goalkeeping_diving = st.slider('GK diving : ', min(data.goalkeeping_diving), max(data['goalkeeping_diving']))
    goalkeeping_handling = st.slider('GK handling : ', min(data.goalkeeping_handling), max(data['goalkeeping_handling']))
    goalkeeping_kicking = st.slider('GK kicking : ', min(data.goalkeeping_kicking), max(data['goalkeeping_kicking']))
    goalkeeping_positioning = st.slider('GK positioning : ', min(data.goalkeeping_positioning), max(data['goalkeeping_positioning']))
    goalkeeping_reflexes = st.slider('GK reflexes : ', min(data.goalkeeping_reflexes), max(data['goalkeeping_reflexes']))
    
    # feauture engineering
    Penaltyist_Score = (0.4 * mentality_penalties + 0.4 * shooting + 0.1 * power_shot_power + 0.1 * mentality_vision) / 4  
    height_to_weight_ratio = height_cm / weight_kg 
    value_to_wage_ratio = value_eur / wage_eur
    attacking_ratio = (attacking_crossing + attacking_finishing 
                         + attacking_heading_accuracy + attacking_short_passing + attacking_volleys)/5
    main_ratio = (pace + shooting + passing + dribbling + defending + physic)/6
    skill_ratio = (skill_dribbling + skill_curve + skill_fk_accuracy + skill_long_passing + skill_ball_control) / 5
    movement_ratio = (movement_acceleration + movement_sprint_speed + movement_agility + movement_reactions + movement_balance)/5
    power_ratio = (power_jumping + power_shot_power + power_stamina + power_strength + power_long_shots) / 5
    mentality_ratio = (mentality_aggression + mentality_interceptions + 
                         mentality_positioning + mentality_vision + mentality_penalties + mentality_composure) / 6
    defending_ratio = (defending_marking_awareness + defending_standing_tackle + defending_sliding_tackle) / 3
    gk_ratio = (goalkeeping_diving + goalkeeping_handling + goalkeeping_kicking + goalkeeping_positioning + goalkeeping_reflexes) / 5
    attack_to_defense_ratio = (attacking_ratio / defending_ratio)
    skill_to_movement_ratio = (skill_ratio / movement_ratio)
    
    attack_to_movement_ratio = (attacking_ratio / movement_ratio)
    power_to_defence_ratio = (power_ratio/ defending_ratio)
    skill_to_defence_ratio = (skill_ratio / defending_ratio)
    attack_to_gk_ratio = (attacking_ratio / gk_ratio)
    

    model_choice = ['neural(one hot encoded)', 'Random Forest Classifier(duplicates)', 'Random Forest Classifier(One hot encoded)']
    choice = st.selectbox('Choose the model: ',  model_choice)

    if choice == 'neural (one hot encoded)':
        pipeline = keras.models.load_model('/app/Neural_tf')
    
    # if choice == 'Random Forest Classifier(duplicates)':
    #     pipeline = joblib.load('/models/RandomForest_for_duplicated_Taget.sav')

    # if choice == 'Random Forest Classifier(One hot encoded)':
    #     pipeline = joblib.load('/models/RandomForest_for_One_Hot_Encoded_Taget.sav')

    # if st.button('Make Prediction'):

    #     if choice == 'neural(one hot encoded target)':
    #         preferred_foot = encoder_foot.transform([preferred_foot])
    #         work_rate = encoder_work.transform([work_rate])
    #         body_type = encoder_body.transform([body_type])

    #         inputs = [overall, age, club_jersey_number,preferred_foot, weak_foot, skill_moves, work_rate, body_type,# 8
    #             pace, shooting, passing, # 11
    #             dribbling, defending, physic, attacking_crossing, # 15
    #             attacking_finishing, attacking_heading_accuracy, # 17
    #             attacking_short_passing, attacking_volleys, skill_dribbling, # 20
    #             skill_curve, skill_fk_accuracy, skill_long_passing, # 23
    #             skill_ball_control, movement_acceleration, movement_sprint_speed, # 26
    #             movement_agility, movement_reactions, movement_balance, # 29
    #             power_shot_power, power_jumping, power_stamina, power_strength, # 33
    #             power_long_shots, mentality_aggression, mentality_interceptions, # 36
    #             mentality_positioning, mentality_vision, mentality_penalties, # 39
    #             mentality_composure, defending_marking_awareness, # 41
    #             defending_standing_tackle, defending_sliding_tackle, # 43
    #             goalkeeping_diving, goalkeeping_handling, goalkeeping_kicking, # 46
    #             goalkeeping_positioning, goalkeeping_reflexes, Penaltyist_Score, # 49
    #             height_to_weight_ratio, value_to_wage_ratio, attacking_ratio, # 52
    #             main_ratio, skill_ratio, movement_ratio, power_ratio, # 56
    #             mentality_ratio, defending_ratio, gk_ratio, # 59
    #             attack_to_defense_ratio, skill_to_movement_ratio, # 61
    #             attack_to_movement_ratio, power_to_defence_ratio, # 63
    #             skill_to_defence_ratio, attack_to_gk_ratio] # 65

            
    #         inputs = np.array(inputs)
    #         inputs = inputs.reshape(1, -1)
    #         scaler.transform(inputs)
    #         pred = predict(inputs, pipeline=pipeline)
    #         pred = pred.round()
    #         columns = ['CAM', 'CB', 'CDM', 'CF', 'CM', 'GK', 'LB', 'LM', 'LW', 'LWB', 'RB',
    #         'RM', 'RW', 'RWB', 'ST']
    #         pred = pd.DataFrame(pred, columns=columns)
    #         st.write(pred)


        # else:
        #     # preferred_foot = le.transform(preferred_foot)
        #     # work_rate = le.transform(work_rate)
        #     # body_type = le.transform(body_type)

        #     inputs = [int(preferred_foot), int(work_rate), int(body_type)]
        #     pred = predict(inputs, pipeline)
        #     st.write(pred)

    st.subheader('Show graphics:')

    graph_choice = ['Top 100 by overall', 'Top 100 by wage', 'Top 30 goalkeepers', 'Top 30 clubs by player overall', 
                   'Top 30 clubs by summary overall', 'Top 30 clubs by avarage pace', 'Top leagues by dribbling', 
                   'Top clubs by top players', 'Player role distribution by age', 'Player positions distribution by age',
                   'Positions distribution by overall', 'Distribution by nationalities']
    
    choice_graph = st.selectbox('Choose the option: ',  graph_choice)

    if st.button('Show'):

        if choice_graph == 'Top 100 by overall':
            out = top_100_overall(data)
            st.write(out)

        elif choice_graph == 'Top 100 by wage':
            out = top_100_wage(data)
            st.write(out)

        elif choice_graph == 'Top 30 goalkeepers':
            out = top_30_gk(data)
            st.write(out)

        elif choice_graph == 'Top 30 clubs by player overall':
            out = top_30_clubs_by_player_overall(data)
            st.write(out)

        elif choice_graph == 'Top 30 clubs by summary overall':
            out = top_30_clubs_sum_players_overall(data)
            st.write(out)

        elif choice_graph == 'Top 30 clubs by avarage pace':
            out = top_30_clubs_avg_pace(data)
            st.write(out)

        elif choice_graph == 'Top leagues by dribbling':
            out = top_leagues_dribbling(data)
            st.write(out)

        elif choice_graph == 'Top clubs by top players':
            out = top_teams_by_top_players(data)
            st.write(out)

        elif choice_graph == 'Player role distribution by age':
            distribution_by_role_age(data_duplicated)
            st.image('/app/Graphs/Distribution_by_role.png')

        elif choice_graph == 'Player positions distribution by age':
            distribution_by_age_pos(data_duplicated)
            st.image('/app/Graphs/Розподіл віку гравців за позиціями.png')

        elif choice_graph == 'Positions distribution by overall':
            distribution_by_overall_pos(data_duplicated)
            st.image('/app/Graphs/Розподіл рейтингу гравців за позиціями.png')

        elif choice_graph == 'Distribution by nationalities':
            dis_by_nationalities
            st.image('/app/Graphs/Розподіл рейтингу гравців за національністю.png')

    # if choice == 'neural (one hot encoded)':

    # pipeline = keras.models.load_model('/app/Neural_tf')
    preferred_foot = encoder_foot.transform([preferred_foot])
    st.write(preferred_foot)
    work_rate = encoder_work.transform([work_rate])
    body_type = encoder_body.transform([body_type])


    inputs = [overall, age, club_jersey_number, int(preferred_foot), weak_foot, skill_moves, int(work_rate), int(body_type),# 8
                pace, shooting, passing, # 11
                dribbling, defending, physic, attacking_crossing, # 15
                attacking_finishing, attacking_heading_accuracy, # 17
                attacking_short_passing, attacking_volleys, skill_dribbling, # 20
                skill_curve, skill_fk_accuracy, skill_long_passing, # 23
                skill_ball_control, movement_acceleration, movement_sprint_speed, # 26
                movement_agility, movement_reactions, movement_balance, # 29
                power_shot_power, power_jumping, power_stamina, power_strength, # 33
                power_long_shots, mentality_aggression, mentality_interceptions, # 36
                mentality_positioning, mentality_vision, mentality_penalties, # 39
                mentality_composure, defending_marking_awareness, # 41
                defending_standing_tackle, defending_sliding_tackle, # 43
                goalkeeping_diving, goalkeeping_handling, goalkeeping_kicking, # 46
                goalkeeping_positioning, goalkeeping_reflexes, Penaltyist_Score, # 49
                height_to_weight_ratio, value_to_wage_ratio, attacking_ratio, # 52
                main_ratio, skill_ratio, movement_ratio, power_ratio, # 56
                mentality_ratio, defending_ratio, gk_ratio, # 59
                attack_to_defense_ratio, skill_to_movement_ratio, # 61
                attack_to_movement_ratio, power_to_defence_ratio, # 63
                skill_to_defence_ratio, attack_to_gk_ratio] # 65
    
    pipeline = keras.models.load_model('/app/Neural_tf')
    # pipeline = keras.models.load_model('/Users/home/DataScience/coursePr/Neural_tf')

    inputs = np.array(inputs)
    inputs = inputs.reshape(1, -1)
    scaler.transform(inputs)
    pred = predict(inputs, pipeline=pipeline)
    columns = ['CAM', 'CB', 'CDM', 'CF', 'CM', 'GK', 'LB', 'LM', 'LW', 'LWB', 'RB',
            'RM', 'RW', 'RWB', 'ST']
    pred = pd.DataFrame(pred, columns=columns)
    st.write(pred)


if __name__ == '__main__':
    main()
