add_library(test OBJECT 
    random_game.cpp human_player_test.cpp play.cpp
    beale_function.cpp sphere_function.cpp rastrigin_function.cpp
    cem_test.cpp nelder_mead_test.cpp variability_timer_multithreads.cpp
    tournament.cpp
)
target_include_directories(test PUBLIC .)

include_directories(../misc ../game ../player ../ui ../optimization)
