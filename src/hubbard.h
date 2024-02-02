#ifndef HUBBARD_H

#include <iostream>

#define NOMINMAX
#include <span>
#include <algorithm>
#include <thread>
#include <mutex>
#include <semaphore>
#include <future>
#include <chrono>
#include <concepts>
#include <variant>
#include <utility> 
#include <string_view>
#include <unordered_map>
using namespace std::chrono_literals;

#ifdef _WIN32
#include <Windows.h>
#include <timeapi.h>
#else
#include <unistd.h>
#endif

#include <imgui.h>
#include <implot.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include <GLFW/glfw3.h>

#include "common.h"
#include "allocator.h"
#include "utils.h"
#include "basis.h"
#include "compute.h"
#include "solver.h"
#include "profiler.h"
#include "ui.h"

#define HUBBARD_H
#endif
