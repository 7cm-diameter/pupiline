# Load required packages
if (!require("tidyverse")) {
  install.packages("tidyverse")
  library(tidyverse)
}

if (!require("comprexr")) {
  if (require("devtools")) {
    devtools::install_github("7cm-diameter/comprexr")
  } else {
    install.packages("devtools")
    devtools::install_github("7cm-diameter/comprexr")
  }
  library(comprexr)
}

# Settings for data alignment
FPS <- 30
TIMEWINDOW <- 3

# Aligning raw data
aligned_data <- list.files("data/area", full.names = T, pattern = "csv") %>%
  lapply(., function(path) {
    read.csv(path) %>%
      add_metadata_to_df(., path) %>%
      mutate(onset = diff(c(0, cs)), frame = seq_len(nrow(.))) %>%
      align_with(., "onset", 1, "frame", -TIMEWINDOW * FPS, TIMEWINDOW * FPS)
  }) %>%
  do.call(rbind, .)

# Plot aligned data
ggplot(aligned_data) +
  geom_line(aes(x = frame / FPS, y = Pupil.area, color = as.factor(serial)),
    alpha = 0.1, size = 1
  ) +
  stat_summary(
    fun.data = "mean_se", geom = "ribbon", aes(x = frame / FPS, y = Pupil.area),
    size = 2, alpha = .25
  ) +
  stat_summary(
    fun = "mean", geom = "line", aes(x = frame / FPS, y = Pupil.area),
    size = 2
  ) +
  geom_vline(xintercept = 0, linetype = "dashed", size = 2) +
  ylab("Pupil size") +
  xlab("Time from event onset") +
  theme_classic() +
  theme(aspect.ratio = .75, legend.position = "none")
