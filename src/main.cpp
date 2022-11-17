#include "faceApp.h"

bool test(std::string filename, std::vector<std::string> &name, std::vector<std::vector<std::string>> &info)
{
	std::ifstream ifs{filename};
	if (!ifs.is_open())
	{
		std::cerr << "Could not open file for reading!\n";
		return EXIT_FAILURE;
	}
	IStreamWrapper isw{ifs};

	Document doc{};
	doc.ParseStream(isw);

	const Value &content = doc["stream"];

	for (int i = 0; i < content.Size(); i++)
	{
		name.push_back(content[i]["camera_id"].GetString());
		std::vector<std::string> value;
		value.push_back(content[i]["address"].GetString());
		value.push_back(content[i]["encode_type"].GetString());
		value.push_back(content[i]["type"].GetString());
		info.push_back(value);
	}
	return EXIT_SUCCESS;
}

int main(int argc, char *argv[])
{
	QDTLog::init("logs/");

	FaceApp app(argc, argv);
	app.init();
	app.run();

	return 0;

	// std::vector<std::string> name;
	// std::vector<std::vector<std::string>> info;
	// test("../configs/video_list.json", name, info);
}
