#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"

#include <fstream>
#include <iostream>

using namespace rapidjson;

auto main(int argc, char **argv) -> int {

  ifstream if("context.json");

  std::string json((istreambuf_iterator<char>(if)),
		   istreambuf_iterator<char>());

  Document doc;

  doc.Parse(if.c_str());

  if (doc.HasParseError()) {
    std::cerr << "Error parsing json: " 
	      << doc.GetParseError() << std::endl;
  }

  return 0;
}

