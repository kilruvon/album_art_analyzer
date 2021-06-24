import musicbrainzngs as mb
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MultiLabelBinarizer

import pandas as pd
from array import array
import os
from PIL import Image
import sys
import time
import ast
import pickle
import colored
from colored import stylize
from urllib.error import HTTPError


mb.set_useragent("")
mb.auth('')

subscription_key = ""
endpoint = ""
computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))


#===========================================================ALBUM DATA PARSER=============================================================================================
#Gives each genre from the genre_list.txt a unique id to work with
def set_genres_id():
    genre_list_file = open('genre_list.txt', 'r', encoding="utf8")
    genre_list = genre_list_file.read().split(sep=',')
    genres_dict = {genre_list[i]:i for i in range(0, len(genre_list))}
    return genres_dict

#Input: release group id (str). 
#Output:
#bool 'hasGenres' - indicates whether vaild genres were found.
#list 'genres' of valid genres found.
def get_genres(rg_id):
    genre_list_file = open('genre_list.txt', 'r', encoding="utf8")
    genre_list = genre_list_file.read().split(sep=',')
    track_id = ''
    tags = []
    release_group = mb.get_release_group_by_id(rg_id, includes = ['tags', 'releases'])
    if 'tag-list' in release_group['release-group']:
        print('found tag-list in release-group')
        for tag in release_group['release-group']['tag-list']:
            tags.append(tag['name'])
    else:
        id = release_group['release-group']['release-list'][0]['id']

        release = mb.get_release_by_id(id, includes = ['tags', 'recordings'])
        if 'tag-list' in release['release']:
            print('found tag-list in release')
            for tag in release['release']['tag-list']:
                tags.append(tag['name'])
        else:
            for track in release['release']['medium-list'][0]['track-list']:
                if track['position'] == '1':
                    track_id = track['recording']['id']
                    break
            recording = (mb.get_recording_by_id(track_id, includes='tags'))
            if 'tag-list' in recording['recording']:
                print('found tag-list in first recording')
                for tag in recording['recording']['tag-list']:
                    tags.append(tag['name'])
    genres = []
    hasGenres = True
    for tag in tags:
        if tag in genre_list:
            genres.append(tag)
    if genres == []:
        hasGenres = False
        print('SKIPPED! (found no genres)')
    return hasGenres, genres


#Input: release group id (str). 
#Output:
#bool 'hasCover' - indicates whether the album art was found.
#str 'coverType' - indicates source of the cover ('RG' - release group, 'R' - release).
#str 'art_title' - the id of the art source.
def get_cover(rg_id):
    art_title = None
    hasCover = True
    coverType = None
    release_group = mb.get_release_group_by_id(rg_id, includes = ['releases'])
    if ('cover-art-archive' in release_group['release-group']) and (release_group['release-group']['cover-art-archive']['artwork'] == 'true'):
        print('found cover art in the release-group')
        coverType = 'RG'
        art_title = rg_id
    else:
        id = release_group['release-group']['release-list'][0]['id']
        release = mb.get_release_by_id(id, includes = ['recordings'])
        if ('cover-art-archive' in release['release']) and (release['release']['cover-art-archive']['artwork'] == 'true'):
            print('found cover art in release')
            art_title = id
            coverType = 'R'
        else:
            print('SKIPPED! (found no cover art)')
            hasCover = False
            art_title = None
    if art_title == None:
            hasCover = False
    return hasCover, coverType, art_title


#Input: art_title (str), coverType (str) - both are received from get_cover(). 
#Saves the album art as .jpg file.
def save_cover(art_title, coverType):
    if coverType == 'RG':
        if (mb.get_release_group_image_list(art_title)['images'][0]['front'] != False):
            data = mb.get_release_group_image_front(art_title, size='500')
        else:
            raise ValueError ('I have an image, but it is not a front one?')
    elif coverType == 'R':
        if (mb.get_image_list(art_title)['images'][0]['front'] != False):
            data = mb.get_image_front(art_title, size='500')
        else:
            data = mb.get_image(art_title, mb.get_image_list(art_title)['images'][0]['id'], size='500')
    with open('database/{}.jpg'.format(art_title), 'wb') as file:
        file.write(data)


#Fetches 1000 albums from the database.
#IDs and genres are saved as database/albums.txt.
def get_albums():
    off = 0
    counter = 0
    while len(os.listdir('database'))-1 < 5000:
        off = str(off)
        result = mb.search_release_groups(limit='100', offset = off, type = 'Album')
        for releasegroup in result['release-group-list']:
            if len(os.listdir('database'))-1 < 5000:
                print('Attempt № ', counter)
                rg_id = releasegroup['id']
                counter += 1
                try:
                    hasGenres, current_genres = get_genres(rg_id)
                except (IndexError, TimeoutError, HTTPError, mb.NetworkError, UnicodeEncodeError, mb.ResponseError) as e:
                    print('There was an error, but I handled that')
                    pass
                else:
                    if hasGenres:
                        try:
                            hasCover, coverType, art_title = get_cover(rg_id)
                        except (IndexError, TimeoutError, HTTPError, mb.NetworkError, mb.ResponseError) as e:
                            print('There was an error, but I handled that')
                            pass
                        else:
                            if hasGenres and hasCover:
                                try:
                                    save_cover(art_title, coverType)
                                except (IndexError, TimeoutError, HTTPError, mb.NetworkError, mb.ResponseError) as e:
                                    print('There was an error, but I handled that')
                                    pass
                                else:
                                    print('SUCCESS! Fetched ',len(os.listdir('database'))-1,' albums already.')
                                    with open('database/albums.txt', 'a', encoding="utf8") as file:
                                        try:
                                            file.write(str(art_title)+'.jpg:'+str(current_genres)+'\n')
                                        except UnicodeEncodeError as e:
                                            print('There was an error, but I handled that')
                                            pass
                                        else:
                                            print(art_title,'.jpg:',current_genres,'\n')
        off = int(off)
        off += 100          


#Removes duplicates or any extra info/covers from the database.
def validate_files():
    fixed_album_list = []
    extra_covers_list = []
    album_ids_list = []
    album_list_file = open('database/albums.txt', 'r', encoding="utf8")
    album_list = album_list_file.read().split(sep='\n')[:-1]
    print('There are currently',len(album_list),' album infos')
    cover_list = os.listdir('database')
    cover_list.remove('albums.txt')
    if 'data.txt' in cover_list:
        cover_list.remove('data.txt')
    print('There are currently', len(cover_list), ' album covers')
    print('Fixing errors..')
    album_list = list(dict.fromkeys(album_list))
    for album in album_list:
        if album[:40] in cover_list:
            fixed_album_list.append(album)
            album_ids_list.append(album[:40])
    for cover in cover_list:
        if cover not in album_ids_list:
            extra_covers_list.append(cover)
    print('There are now',len(fixed_album_list),' album infos')
    print('extra_covers_list: ', extra_covers_list)
    with open('database/albums.txt', 'w', encoding="utf8") as file:
        for album in fixed_album_list:
            file.write(album+'\n')
        file.write
    path = "database/"
    for cover in extra_covers_list:
        full_file_name = os.path.join(path,cover)
        os.remove(full_file_name)
    control_cover_list = os.listdir('database')
    control_cover_list.remove('albums.txt')
    if 'data.txt' in control_cover_list:
        control_cover_list.remove('data.txt')
    if (set(album_ids_list) == set(control_cover_list)) and (len(album_ids_list) == len(control_cover_list)):
        print('All errors are fixed!')
    else:
        sys.exit('Something is still wrong!')
    print('There are currently ',len(fixed_album_list),' album infos and ',len(control_cover_list),' album covers.')

#==========================================================ALBUM COVER ANALYZER=======================================================================================================

#Input: name of the album cover file in database/ (str). 
#Output:
#list 'tags' - contains objects that Microsoft Azure's Computer Vision found on the cover art.
#list 'colors' - contains dominant colors of the cover art.
def describe_cover(cover_file_name):
    tags = []
    unwanted_tags = ['text', 'screenshot', 'poster', 'book', 'cartoon', #here we can remove tags that appear too often and are irrelevant
    'illustration', 'design', 'graphic', 'person', 'human face',
    'clothing', 'man', 'music', 'drawing', 'art', 'sign']                                    
    image = open('database/'+ cover_file_name, "rb")
    result = computervision_client.analyze_image_in_stream(image, visual_features = ['tags','color'])
    for tag in result.tags:
        if tag.name not in unwanted_tags: 
            tags.append(tag.name)
    colors = (result.color.dominant_colors)
    if result.color.accent_color not in colors:
        colors.append(result.color.accent_color)
    return tags, colors


#Creates data.txt file that contains objects, colors and genres for each album.
def get_descriptions():
    album_list_file = open('database/albums.txt', 'r', encoding="utf8")
    album_list = album_list_file.read().split(sep='\n')[:-1]
    files = os.listdir('database')
    if 'data.txt' not in files:
        with open('database/data.txt', 'w') as file:
            for album in album_list:
                tags, colors = describe_cover(album[:40])
                print(str(tags)+':'+str(colors)+':'+str(album[41:]))
                file.write(str(tags)+':'+str(colors)+':'+str(album[41:])+'\n')

#================================================================IDEA GENERATOR=================================================================================

#This function fills the genre dictionaries with tags
def dict_filler(result_dict, genre, list_to_append):
    if genre not in result_dict:
                result_dict[genre] = []
    result_dict[genre].append(list_to_append)

#This function transforms genre dictionaries into lists
def fit_transformer(dictionary, mlb_mode):
    genre_ids = set_genres_id()
    result_list = []
    for item in dictionary:
        dictionary[item] = mlb_mode.fit_transform(dictionary[item])
    for item in dictionary:
        for elem in dictionary[item]:
            elem = list(elem)
            elem.insert(0, genre_ids[item])
            result_list.append(elem)
    return result_list

#This function splits the dataframe into two parts to feed it into ML algorithm
def dataframe_splice(df):
    x = df['GENRE']
    y = df.drop('GENRE', axis = 1)
    return x, y

#Converts data.txt into a DataFrame object
#Output:
#series/dataframe 'x_tags', 'y_tags', 'x_colors', 'y_colors' - spliced DataFrames for each type of data
#MultiLabelBinarizer 'mlb_tags', 'mlb_colors' - sets of binarized values
def dataframe_convert():
    unique_tags = []
    unique_colors = []
    result_dict_tags = {}
    result_dict_colors = {}
    data_file = open('database/data.txt', 'r', encoding="utf8")
    data_list = data_file.read().split(sep='\n')[:-1]
    for line in data_list:
        temp_tag_list = ast.literal_eval(line.split(":")[0])
        temp_color_list = ast.literal_eval(line.split(":")[1])
        temp_genre_list = ast.literal_eval(line.split(":")[2])
        for tag in temp_tag_list:
            if tag not in unique_tags:
                unique_tags.append(tag)
        for color in temp_color_list:
            if color not in unique_colors:
                unique_colors.append(color)

        for genre in temp_genre_list:
            dict_filler(result_dict_tags, genre, temp_tag_list)
            dict_filler(result_dict_colors, genre, temp_color_list)

    mlb_tags = MultiLabelBinarizer(classes = sorted(unique_tags))
    mlb_colors = MultiLabelBinarizer(classes = sorted(unique_colors))

    result_list_tags = fit_transformer(result_dict_tags, mlb_tags)
    tags_columns = list(mlb_tags.classes_)
    tags_columns.insert(0, "GENRE")
    result_list_colors = fit_transformer(result_dict_colors, mlb_colors)
    colors_columns = list(mlb_colors.classes_)
    colors_columns.insert(0, "GENRE")

    df_tags = pd.DataFrame.from_records(result_list_tags, columns=tags_columns)
    df_colors = pd.DataFrame.from_records(result_list_colors, columns=colors_columns)

    x_tags, y_tags = dataframe_splice(df_tags)
    x_colors, y_colors = dataframe_splice(df_colors)
    return x_tags, y_tags, x_colors, y_colors, mlb_tags, mlb_colors

#This function receives the user input and returns a dataframe element, made from it
def user_input_to_df():
    genre = input("Enter genre name:")
    genre_ids = set_genres_id()
    if genre in genre_ids:
        found_genres = []
        data_file = open('database/data.txt', 'r', encoding="utf8")
        data_list = data_file.read().split(sep='\n')[:-1]
        for line in data_list:
            temp_genre_list = ast.literal_eval(line.split(":")[2])
            for item in temp_genre_list:
                if item not in found_genres:
                    found_genres.append(item)
        if genre in found_genres:
            genre_id = genre_ids[genre]
            df = pd.DataFrame([genre_id])
        else:
            sys.exit('No such genre found in the given dataset!')
    else:
        sys.exit('No such genre found (in general)! Refer to genre-list.txt')
    return df

#Input: x and y parts of the dataframes - received from dataframe_convert(). 
#Uses the RandomForestRegressor algorithm and trains two instances of it.
#Saves the results as 'model_tags.pkl' and 'model_colors.pkl' using Pickle
def machine_learning(x_tags, y_tags, x_colors, y_colors):
    model_tags = RandomForestRegressor()
    model_tags.fit(x_tags.values.reshape(-1, 1), y_tags)
    model_colors = RandomForestRegressor()
    model_colors.fit(x_colors.values.reshape(-1, 1), y_colors)
    with open('model_tags.pkl','wb') as f:
        pickle.dump(model_tags,f)
    with open('model_colors.pkl','wb') as f:
        pickle.dump(model_colors,f)

def main():
    get_albums()                                               
    #Took ~24 hours and scanned 21320 albums in total to get 1000 valid albums.
    #Took around 96 hours to get the remaining 4000 albums with 110000 attempts.
    #(that means only 4.7% of albums had the required data)

    validate_files()

    get_descriptions()

    x_tags, y_tags, x_colors, y_colors, mlb_tags, mlb_colors = dataframe_convert()

    if not os.path.isfile('model_tags.pkl') or not os.path.isfile('model_colors.pkl'):
        machine_learning(x_tags, y_tags, x_colors, y_colors)

    with open('model_tags.pkl', 'rb') as f:
        model_tags = pickle.load(f)

    with open('model_colors.pkl', 'rb') as f:
        model_colors = pickle.load(f)

    AzureColors = {'Black':'#000000', 'Blue':'#0000FF', 'Brown':'#8B4513', 
    'Green':'#00FF00', 'Grey':'#808080', 'Orange':'#FF8C00', 'Pink':'#FFC0CB', 
    'Purple':'#8A2BE2', 'Red':'#FF0000', 'Teal':'#008080', 'White':'#FFFFFF', 'Yellow':'#FFFF00'}

    while(True):
        df = user_input_to_df()
        res_tags = dict(zip(mlb_tags.classes_,model_tags.predict(df)[0]))
        res_colors = dict(zip(mlb_colors.classes_,model_colors.predict(df)[0]))
        top_tags = sorted(res_tags, key=res_tags.get, reverse=True)[:10]
        top_colors = sorted(res_colors, key=res_colors.get, reverse=True)[:10]
        print('Matching tags:', top_tags)
        print('Matching colors:', end =" ")
        for color in top_colors:
            if color in AzureColors:
                print(stylize(color, colored.fg(AzureColors[color])), end =" ")
            else:
                print(stylize(color, colored.fg('#'+color)), end =" ")
        print('\n')

if __name__ == '__main__':
    main()