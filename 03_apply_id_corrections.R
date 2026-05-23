library(readr)
library(dplyr)

cfg_all <- read_csv("Data/task/cfg_all_with_g_alpha.csv")
id_corrections <- read_csv("Data/task/id_corrections.csv")

#remove test participants
ids_to_drop <- id_corrections %>%
  filter(action == "drop") %>%
  pull(ID)

cfg_all <- cfg_all %>%
  filter(!ID %in% ids_to_drop)

#fix id 
id_fixes <- id_corrections %>%
  filter(action == "fix") %>%
  select(ID, new_ID)

cfg_all <- cfg_all %>%
  left_join(id_fixes, by = "ID") %>%
  mutate(ID = if_else(!is.na(new_ID), new_ID, ID)) %>%
  select(-new_ID)

#remove duplicates
cfg_all <- cfg_all %>%
  distinct()

write_csv(cfg_all, "Data/task/cfg_all_ready_for_merge.csv")