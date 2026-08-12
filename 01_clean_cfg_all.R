library(readr)
library(dplyr)

# add column for imputed steps count
cfg_all <- read_csv("Data/task/cfg_all.csv")
imputed_shapes_count <- read_csv("Data/task/before_imputation/imputed_shapes_count.csv")

cfg_all <- cfg_all %>%
  left_join(
    imputed_shapes_count %>% select(ID, imputed_moves),
    by = "ID"
  ) %>%
  rename(imputed_steps_count = imputed_moves)

# calculate percent of imputed steps out of total moves
cfg_all <- cfg_all %>%
  mutate(imputed_steps_percent = (imputed_steps_count / `Total # moves`) * 100)

# flag participants with more than 5 percent imputed steps
cfg_all <- cfg_all %>%
  mutate(imputed_over_5 = imputed_steps_percent > 5)

write_csv(cfg_all, "Data/task/cfg_all_clean.csv")
