## rosebud ##
""" 
rosebud is a tool for pulling CSVs into python, and immediately extracting basic statistics (mean, median, mode, quartile
data, etc.) into variables, and has the capability of plotting these preliminary statistics via seaborn pairplots.
""";

#INIT BLOCK - place imports, initializations, functions and helper tools here:

#imports:
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import warnings

#initializations:
# - global list of table names:
rosebud_tas_tables = []

#helpers:
# - listReplace. If character is in provided list of chars to replace, replace with replacement.:
def listReplace(targetString, listSelection, replacement):
    for char in listSelection:
        if char in targetString:
            targetString = targetString.replace(char, replacement)
    return  targetString

# - Heatmap mapper: group feature columns names with specific levels of correlation:
def hMapper(df):
    df = pd.DataFrame(df)
    df = df.corr()
    
    StrPos_init = []
    WkPos_init = []
    NoCorr_init = []
    WkNeg_init = []
    StrNeg_init = []
    
    StrPos = []
    WkPos = []
    NoCorr = []
    WkNeg = []
    StrNeg = []
    
    StrPos_cp = []
    WkPos_cp = []
    NoCorr_cp = []
    WkNeg_cp = []
    StrNeg_cp = []
    
    features = list(df.columns)
    for i in range(0, len(features)):
        for j in range(i, len(features)):
            if df.iloc[j,i] == 1:
                continue
            else:
                if df.iloc[j,i] > 0.7:
                    StrPos_init.append(features[i])
                    StrPos_init.append(df.index[df[features[i]] == df.iloc[j,i]][0])
                    StrPos_cp.append([features[i], df.index[df[features[i]] == df.iloc[j,i]][0]])
                    
                elif df.iloc[j,i] > 0.3 and df.iloc[j,i] <= 0.7:
                    WkPos_init.append(features[i])
                    WkPos_init.append(df.index[df[features[i]] == df.iloc[j,i]][0])
                    WkPos_cp.append([features[i], df.index[df[features[i]] == df.iloc[j,i]][0]])
                    
                elif df.iloc[j,i] > -0.3 and df.iloc[j,i] <= 0.3:
                    NoCorr_init.append(features[i])
                    NoCorr_init.append(df.index[df[features[i]] == df.iloc[j,i]][0])
                    NoCorr_cp.append([features[i], df.index[df[features[i]] == df.iloc[j,i]][0]])
                    
                elif df.iloc[j,i] >= -0.7 and df.iloc[j,i] <= -0.3:
                    WkNeg_init.append(features[i])
                    WkNeg_init.append(df.index[df[features[i]] == df.iloc[j,i]][0])
                    WkNeg_cp.append([features[i], df.index[df[features[i]] == df.iloc[j,i]][0]])
                    
                elif df.iloc[j,i] < -0.7:
                    StrNeg_init.append(features[i])
                    StrNeg_init.append(df.index[df[features[i]] == df.iloc[j,i]][0])
                    StrNeg_cp.append([features[i], df.index[df[features[i]] == df.iloc[j,i]][0]])

    
    [StrPos.append(val) for val in StrPos_init if val not in StrPos]
    [WkPos.append(val) for val in WkPos_init if val not in WkPos]
    [NoCorr.append(val) for val in NoCorr_init if val not in NoCorr]
    [WkNeg.append(val) for val in WkNeg_init if val not in WkNeg]
    [StrNeg.append(val) for val in StrNeg_init if val not in StrNeg]
    
    return StrPos, WkPos, NoCorr, WkNeg, StrNeg, StrPos_cp, WkPos_cp, NoCorr_cp, WkNeg_cp, StrNeg_cp

def normalizer(data):
    for col in data.columns:
        if data[col].dtype != 'object':
            data[col] = (data[col] - np.mean(data))/np.std(data[col])
        else:
            continue

#suppressing errors:
# - this is to suppress DataFrame NaN value warnings later, which will be handled by a more user-friendly print statement.
warnings.simplefilter("ignore")

def tablesandstats(filepath, show_plots = 'all'):
    """ import the data into the workspace, and produce basic info, as well as an overview of the correlations"""
    
    ok_vals_tas = ['all', 'none', 'heatmap', 'completeness']
    
    if show_plots not in ok_vals_tas:
        print("'show_plots' argument is invalid. Valid arguments are 'all', 'none', 'heatmap' and 'completeness'.")
        return

    #OS AGNOSTIC
    orig_dir = os.getcwd()
    env_folder, tgt_file = os.path.split(filepath)
    os.chdir(env_folder)

    #import data:
    data = pd.DataFrame(pd.read_csv(os.path.join(env_folder, tgt_file)))
    csv_name = tgt_file[:-4]
    csv_var = listReplace(csv_name, ['-',' '], '_')
    csv_title = listReplace(csv_name, ['-','_'], ' ')

    print("\n" + "Rosebud is creating tables and statistics from " + csv_title + "...\n")
    
    # summary of columns + value indexing:
    data_summary = data.describe()
    print("Measures of Center and Basic Descriptive Statistics of " + csv_title + ":", "\n", data_summary)
    data_summary = data.describe()
    idx_names = list(data_summary.index.values)
    idx_col_names = list(data_summary.columns.values)

    #looping for variable name generation:
    stat_names = []
    stat_vals = []
    for j in idx_col_names:
        for i in idx_names:
            stat_names.append(j+"_"+i)
            stat_vals.append(data_summary[j][i])

    var_dict = dict(zip(stat_names, stat_vals))
    globals().update(var_dict)

    #normalize non-object data:
    dataZNorm = data.copy()
    dataZNorm = normalizer(dataZNorm)

    #return dataframes:
    frame_names = [csv_var, csv_var + "_Normalized"]
    frame_contents = [data, dataZNorm]

    frame_intermed = dict(zip(frame_names, frame_contents))
    globals().update(frame_intermed)


    #printing graphs:
    museum = os.listdir()
    graphs = [csv_var + "_correlation_hmap.jpg", csv_var + "_data_completeness.jpg"]

    graphs_in_museum = all(item in museum for item in graphs)

    if graphs_in_museum:
        print("\n!! Charts for " + csv_title + " already present in current directory !!\n")

        # TODO: Show the chart using the image instead of plot show.

    else:
        # data types of columns:
        print("\nFeature Data Types of " + csv_title + ":\n" + data.dtypes.to_string() + "\n")

        if show_plots == 'all':

            #correlation grid graphs:
            print("\nFeature Corrrelations of " + csv_title + ":\n")
            plt.figure(figsize = [20,15])
            corr = data.corr()
            mask = np.zeros_like(corr, dtype=np.bool)
            mask[np.triu_indices_from(mask)] = True
            sns.heatmap(corr,mask = mask, annot=True, cmap = 'gnuplot')
            plt.savefig(csv_var + "_correlation_hmap.jpg",format = 'jpg', dpi=100);
            plt.show();

            #data completeness:
            print("\nDataset completeness of " + csv_title + ":\n");
            print("- " + csv_title + " missing value ratio (percentage):\n");
            print(((data.isnull().sum()/len(data))*100).to_string())
            msno.matrix(data, color = (0.38,0,0.15))
            print("\n- Visual representation of missing value ratio:")
            plt.savefig(csv_var + "_data_completeness.jpg",format = 'jpg', dpi=100);
            plt.show();

        elif show_plots == "heatmap":

            #correlation grid graphs:
            print("\nFeature Corrrelations of " + csv_title + ":\n");
            plt.figure(figsize = [20,15])
            corr = data.corr()
            mask = np.zeros_like(corr, dtype=np.bool)
            mask[np.triu_indices_from(mask)] = True
            sns.heatmap(corr, mask = mask, annot=True, cmap = 'gnuplot');
            plt.savefig(csv_var + "_correlation_hmap.jpg",format = 'jpg', dpi=100);
            plt.show();

            #data completeness:
            print("\nDataset completeness of " + csv_title + ":\n");
            print("- " + csv_title + " missing value ratio (percentage):\n");
            print(((data.isnull().sum()/len(data))*100).to_string())
            msno.matrix(data, color = (0.38,0,0.15));
            plt.savefig(csv_var + "_data_completeness.jpg",format = 'jpg', dpi=100);
            plt.close();

        elif show_plots == "none":

            #correlation grid graphs:
            plt.figure(figsize = [20,15])
            corr = data.corr()
            mask = np.zeros_like(corr, dtype=np.bool)
            mask[np.triu_indices_from(mask)] = True
            sns.heatmap(corr, mask = mask, annot=True, cmap = 'gnuplot')
            plt.savefig(csv_var + "_correlation_hmap.jpg",format = 'jpg', dpi=100);
            plt.close();

            #data completeness:
            print("\nDataset completeness of " + csv_title + ":\n");
            print("- " + csv_title + " missing value ratio (percentage):\n");
            print(((data.isnull().sum()/len(data))*100).to_string())
            msno.matrix(data, color = (0.38,0,0.15));
            plt.savefig(csv_var + "_data_completeness.jpg",format = 'jpg', dpi=100);
            plt.close();

        elif show_plots == "completeness":

            #correlation grid graphs:
            plt.figure(figsize = [20,15])
            corr = data.corr()
            mask = np.zeros_like(corr, dtype=np.bool)
            mask[np.triu_indices_from(mask)] = True
            sns.heatmap(corr, mask = mask, annot=True, cmap = 'gnuplot')
            plt.savefig(csv_var + "_correlation_hmap.jpg",format = 'jpg', dpi=100);
            plt.close();

            #data completeness:
            print("\nDataset completeness of " + csv_title + ":\n");
            print("- " + csv_title + " missing value ratio (percentage):\n");
            print(((data.isnull().sum()/len(data))*100).to_string())
            msno.matrix(data, color = (0.38,0,0.15));
            print("\n- Visual representation of missing value ratio:")
            plt.savefig(csv_var + "_data_completeness.jpg",format = 'jpg', dpi=100);
            plt.show();

    rosebud_tas_tables.append(frame_names[0])
    rosebud_tas_tables.append(frame_names[1])
    print("Tables created:"+ "\n" + "* " + frame_names[0] + "\n* " + frame_names[1])
    os.chdir(orig_dir)
    

"""survey takes in the file path, normalizes data, and performs pair plotting to the columns as determined by correlation
grid, stratified by levels of correlation """
def survey(filepath, filter_by = 'all', regress = False):
    
    ok_vals_pltI = ['all', 'strong_pos', 'weak_pos', 'no_corr', 'weak_neg', 'strong_neg']
    
    if filter_by not in ok_vals_pltI:
        print("'filter_by' argument is invalid. Valid arguments are 'strong_pos', 'weak_pos','no_corr', 'weak_neg', 'strong_neg' and 'all'.")
        return
        
    #get the data:
    orig_dir = os.getcwd()
    env_folder, tgt_file = os.path.split(filepath)
    os.chdir(env_folder)
    data = pd.DataFrame(pd.read_csv(os.path.join(env_folder, tgt_file)))
    csv_name = tgt_file[:-4]
    csv_var = listReplace(csv_name, ['-',' '], '_')
    csv_title = listReplace(csv_name, ['-','_'], ' ')
    
    print("\n" + "Rosebud is surveying out the data in " + csv_title + ' (note: graphical scale is derived from a normalized data set)\n')
    
    if len(data.isna()) > 0:
        print("!! NOTE: " + csv_title + " contains NaN values, which may affect true correlation value !!\n")
        
    #normalize data:
    for col in data.columns:
        if data[col].dtype != 'object':
            data[col] = (data[col] - np.mean(data[col]))/np.std(data[col])
    
    #obtain columns:
    StrPos, WkPos, NoCorr, WkNeg, StrNeg, StrPos_cp, WkPos_cp, NoCorr_cp, WkNeg_cp, StrNeg_cp = hMapper(data)
    
    print("Strong positive correlations:\n")
    print(str(np.asarray(StrPos_cp)) + '\n')
    print("Weak positive correlations:\n")
    print(str(np.asarray(WkPos_cp)) + '\n')
    print("Features with no correlations:\n")
    print(str(np.asarray(NoCorr_cp)) + '\n')
    print("Weak Negative correlations:\n")
    print(str(np.asarray(WkNeg_cp)) + '\n')
    print("Strong Negative correlations:\n")
    print(str(np.asarray(StrNeg_cp)) + '\n')
    
    #graphing:
    if filter_by == 'all':
        print('Pairwise relationship graphs of all features:')
        if regress:
            sns.pairplot(data, kind = 'reg', diag_kind = 'kde');
        else:
            sns.pairplot(data, diag_kind = 'kde');
        plt.savefig(csv_var + "_All_Feature_Correlations.jpg",format = 'jpg', dpi=100);
        plt.xticks(rotation=30);
        plt.show();
        
    elif filter_by == 'strong_pos':
        if len(StrPos) == 0:
            print('No features have strong positive correlations');
            return
        else:
            print('Pairwise relationship graphs of strong positive correlation features:')
            if regress:
                sns.pairplot(data[StrPos], kind = 'reg', diag_kind = 'kde');
            else:
                sns.pairplot(data[StrPos], diag_kind = 'kde');
            plt.savefig(csv_var + "_Strong_Positive_Feature_Correlations.jpg",format = 'jpg', dpi=100);
            plt.xticks(rotation=30);
            plt.show();
        
    elif filter_by == 'weak_pos':
        if len(WkPos) == 0:
            print('No features have weak positive correlations');
            return
        else:
            print('Pairwise relationship graphs of weak positive correlation features:')
            if regress:
                sns.pairplot(data[WkPos], kind = 'reg', diag_kind = 'kde');
            else:
                sns.pairplot(data[WkPos], diag_kind = 'kde');
            plt.savefig(csv_var + "_Weak_Positive_Feature_Correlations.jpg",format = 'jpg', dpi=100);
            plt.xticks(rotation=30);
            plt.show();
            
    elif filter_by == 'no_corr':
        if len(NoCorr) == 0:
            print('All features have potentially significant correlations');
            return
        else:
            print('Pairwise relationship graphs of features with no discernable correlation:')
            if regress:
                sns.pairplot(data[NoCorr], kind = 'reg', diag_kind = 'kde');
            else:
                sns.pairplot(data[NoCorr], diag_kind = 'kde');
            plt.savefig(csv_var + "_Features_With_no_correlations.jpg",format = 'jpg', dpi=100);
            plt.xticks(rotation=30);
            plt.show();
            
    elif filter_by == 'weak_neg':
        if len(WkNeg) == 0:
            print('No features have weak negative correlations');
            return
        else:
            print('Pairwise relationship graphs of weak negative correlation features:')
            if regress:
                sns.pairplot(data[WkNeg], kind = 'reg', diag_kind = 'kde');
            else:
                sns.pairplot(data[WkNeg], diag_kind = 'kde');
            plt.savefig(csv_var + "_Weak_Negative_Feature_Correlations.jpg",format = 'jpg', dpi=100);
            plt.xticks(rotation=30);
            plt.show();
            
    elif filter_by == 'strong_neg':
        if len(StrNeg) == 0:
            print('No features have strong negative correlations');
            return
        else:
            print('Pairwise relationship graphs of strong negative correlation features:')
            if regress:
                sns.pairplot(data[StrNeg], kind = 'reg', diag_kind = 'kde');
            else:
                sns.pairplot(data[StrNeg], diag_kind = 'kde');
            plt.savefig(csv_var + "_Strong_Negative_Feature_Correlations.jpg",format = 'jpg', dpi=100);
            plt.xticks(rotation=30);
            plt.show();
    os.chdir(orig_dir);

def processfolder(folderpath, show_plots = 'all'):
    """perform tablesandstats on all CSVs within a folder"""

    ok_vals_pf = ['all', 'none', 'heatmap', 'pairplot', 'completeness']
    
    if show_plots not in ok_vals_pf:
        print("'show_plots' argument is invalid. Valid arguments are 'all', 'none', 'heatmap', 'pairplot', and 'completeness'.")
        
    else:
        
        print("\n" + "Rosebud is processing folder:" + folderpath + "...")
        
        #save current directory as a variable
        orig_dir = os.getcwd()

        #os.chdir into target folder
        os.chdir(folderpath)

        #create empty names list
        found_files = []

        #loop through folder list, and apply tablesandstats to each file.
        folder_files = list(os.listdir())
        for wanted_file in folder_files:
            if wanted_file.endswith("csv"):
                found_files.append(wanted_file)

        for f_file in found_files:
            tablesandstats(folderpath + '\\' + f_file, show_plots)

        #reset directory back to original dir.
        os.chdir(orig_dir)

        #printing out available tables:
        print("\nFolder processing complete! Here are the data tables available:" + "\n")
        for rosebud_tas_table in rosebud_tas_tables:
            print("* " + rosebud_tas_table)

if __name__ == '__main__':
    runitem = input('paste file/folder directory here: ')
    folder, file = os.path.split(runitem)
    runitem = os.path.join(folder, file)
    if os.path.isfile(runitem):
        tablesandstats(runitem)
    elif os.path.isdir(runitem):
        processfolder(runitem)
    else:
        print("provided file/folder directory is invalid, please check if file/folder directory is correct")

