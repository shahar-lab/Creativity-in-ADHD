library(readr)
library(dplyr)

# load cleaned task data and participant data
cfg_all <- read_csv("Data/task/cfg_all_ready_for_merge.csv")
participant_data <- read_csv("Data/lab_screening/df_for_creativity_all.csv")

# check basic structure of both datasets
nrow(cfg_all)
nrow(participant_data)

names(cfg_all)
names(participant_data)

# check for duplicate IDs in task data
cfg_all %>%
  count(ID) %>%
  filter(n > 1)

# find participants who appear in participant_data but not in cfg_all
participant_only <- participant_data %>%
  distinct(subjectid, .keep_all = TRUE) %>%
  anti_join(
    cfg_all %>% distinct(ID, .keep_all = TRUE),
    by = c("subjectid" = "ID")
  )

# find participants who appear in cfg_all but not in participant_data
cfg_only <- cfg_all %>%
  distinct(ID, .keep_all = TRUE) %>%
  anti_join(
    participant_data %>% distinct(subjectid, .keep_all = TRUE),
    by = c("ID" = "subjectid")
  )

# check mismatch counts
nrow(participant_only)
nrow(cfg_only)

# inspect mismatch tables
View(participant_only)
View(cfg_only)