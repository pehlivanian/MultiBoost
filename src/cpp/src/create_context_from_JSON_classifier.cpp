#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <iostream>

#include "utils.hpp"

using namespace IB_utils;
using namespace ModelContext;

using namespace boost::program_options;

std::string path_(std::string inputString) {
  std::string fileName;
  for (auto it = inputString.rbegin(); it != inputString.rend() && fileName.size() != 24; ++it) {
    if (std::isalnum(*it)) fileName.push_back(*it);
  }
  return fileName;
}

auto main(int argc, char** argv) -> int {
  // For json input config specification, we serialize
  // ofileName1: first run
  // ofileNamen : subsequent runs [serializeDataset = false, serializeLabels = false]

  std::string dataname, ifileName = path_(typeid(Context).name()), ofileName1, ofileNamen;

  options_description desc("Options");
  desc.add_options()("dataname", value<std::string>(&dataname), "dataset name")(
      "ifileName", value<std::string>(&ifileName), "JSON input filename")(
      "ofileName1", value<std::string>(&ofileName1), "JSON output filename1")(
      "ofileNamen", value<std::string>(&ofileNamen), "JSON output filenamen");

  variables_map vm;

  try {
    store(parse_command_line(argc, argv, desc), vm);

    if (vm.count("help")) {
      std::cout << "Context creator helper" << std::endl << desc << std::endl;
    }
    notify(vm);

  } catch (const error& e) {
    std::cerr << "ERROR: " << e.what() << std::endl;
    std::cerr << desc << std::endl;
  }

  Context context{};

  // Serialize
  using CerealT = Context;
  using CerealIArch = cereal::JSONInputArchive;
  using CerealOArch = cereal::JSONOutputArchive;

  boost::filesystem::path fldr{"./"};
  loads<CerealT, CerealIArch, CerealOArch>(context, ifileName, fldr);

  // Run 1 version
  dumps<CerealT, CerealIArch, CerealOArch>(context, ofileName1, fldr);

  // Runs >= 2 version
  context.serializeDataset = false;
  context.serializeLabels = false;

  dumps<CerealT, CerealIArch, CerealOArch>(context, ofileNamen, fldr);

  return 0;
}
