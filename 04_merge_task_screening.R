library(readr)
library(dplyr)

# load cleaned task data and participant data
cfg_all <- read_csv("Data/task/cfg_all_ready_for_merge.csv")
participant_data <- read_csv("Data/lab_screening/df_for_creativity_all.csv")

# check dataset sizes
nrow(cfg_all)
nrow(participant_data)

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

# inspect mismatch tables if needed
View(participant_only)
View(cfg_only)

# check task exclusions for participants missing from cfg_all
exclusions <- read_csv("Data/task/exclusions.csv")

participant_excluded <- participant_only %>%
  left_join(exclusions, by = c("subjectid" = "ID"))

# check how many participant-only cases are explained by exclusions
sum(!is.na(participant_excluded$reason))
sum(is.na(participant_excluded$reason))

# inspect participant-only cases with exclusion reasons
participant_excluded %>%
  select(subjectid, declared_group, cohort, reason) %>%
  View()

# merge task data with participant data for matched IDs only
df_creativity_ADHD <- cfg_all %>%
  inner_join(participant_data, by = c("ID" = "subjectid"))

# save merged dataset
write_csv(df_creativity_ADHD, "Data/df_creativity_ADHD.csv")
