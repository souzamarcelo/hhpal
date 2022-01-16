#include <iostream>
#include <vector>
#include <fstream>
#include <unordered_set>
#include <chrono>
#include <iomanip>
#include <random>
#include <boost/multi_array.hpp>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

namespace po = boost::program_options;
using namespace boost;
using namespace std;

unsigned timeLimit;
string format;
chrono::system_clock::time_point startTime;

#include "rng.hpp"
#include "instance.hpp"
#include "instancedata.hpp"
#include "solution.hpp"

#include "report.hpp"
Report report;

#include "elite.hpp"
#include "localsearch.hpp"
#include "tabusearch.hpp"
#include "recombine.hpp"

void run(Solution& S, Instance& I, double target, chrono::system_clock::time_point startTime) {
    unsigned bsize = 19;
    double gamma = 0.4214;
    unsigned maxstagnate = 1 * I.n;
    unsigned maxsteps = numeric_limits<unsigned>::max();
    BTS bt;
    StepGenerator sg1(14, I.n>=14*9?I.n/9:14);

    recombine::recombiner(I, S, bsize,
                        [&](const Instance& I, Solution& S, const Solution& T) { return recombine::recombine(I, S, T,recombine::bn, gamma);},
                        [&](Solution& S) { return iterated_localsearch(S, [&](Solution& S) {
                                                        return tabusearch(S, bt, [&]() { return max(1.0,1 * I.n / 100.0); }, startTime, target, maxstagnate, maxsteps);},
                        [&](Solution& S) { return perturbLeastLoss(S, sg1, 7); }, startTime, target);},
                        startTime, target);
}

int main(int argc, char *argv[]) {
    po::options_description desc("General options");
    desc.add_options()
            ("help", "show help")
            ("ins", po::value<string>(), "instance")
            ("seed", po::value<unsigned>()->default_value(0), "random seed")
            ("timelimit", po::value<int>(), "time limit")
            ("timescale", po::value<double>()->default_value(1), "time scale for experiments")
            ;
    
    po::positional_options_description pod;
    pod.add("ins", 1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).positional(pod).run(), vm);
    po::notify(vm);

    cout << fixed;

    if (vm.count("help") || !vm.count("ins")) {
        cout << desc << endl;
        return 0;
    }

    Instance I;
    double target;
    double scale;
    string instanceName;
    bool maximize;
    unsigned seed;

    scale = vm["timescale"].as<double>();
    seed = vm["seed"].as<unsigned>();
    setupRandom(seed);

    instanceName = vm["ins"].as<string>();
    ifstream ins(instanceName);
    if (!ins.good()) {
        cout << "Can't open instance file " << instanceName << endl;
        return 1;
    }

    target = getBestKnownValue(instanceName);
    timeLimit = getTimeLimit(instanceName, scale);
    maximize = getMaximize(instanceName);
    format = getFormat(instanceName);
    target = numeric_limits<int>::min();
    
    if(vm.count("timelimit")) {
        timeLimit = vm["timelimit"].as<int>() * scale;
    }

    I.readInstance(ins, maximize, format);
    Solution S(I);
    startTime = chrono::system_clock::now();
    
    run(S, I, target, startTime);
}