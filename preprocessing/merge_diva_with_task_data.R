library(tidyverse)


#### merge diva with task ---
# load diva results
load('data/diva_after_exclusions_with_sumscores.rdata')
diva = df
rm(df)


#duplicates
#diva |> group_by(subjectid) |> filter(n() > 1) |> ungroup() |> View()

#load task
task = read.csv('data/data_raw/task_with_g_and_alpha.csv')


task <- task |> mutate(subjectid = str_trim(ID)) |> select(-ID)

#duplicates
#task |> group_by(subjectid) |> filter(n() > 1) |> ungroup() |> View()

df = merge(task,diva, by = c('subjectid'),all.y = F)

x = task$subjectid[task$subjectid %in% diva$subjectid == F]
x = df$subjectid[df$subjectid %in% task$subjectid == F]

#### comments for correction of subjects id
#ywsSnI should have lower case L instead of I. fixed manually in the task file
#"52MCE2"   "p3juAg"   "KHWkGA" - 1 childhood dysfunction domain


save(df,file = 'data/df_all.rdata')
write.csv(df,file = 'data/df_all.csv')
