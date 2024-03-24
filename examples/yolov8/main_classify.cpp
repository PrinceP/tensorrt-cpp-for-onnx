#include <cuda_runtime.h>
#include "src_net.h"
#include "crop_resize.h"

#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/pattern_formatter.h>

struct Object
{
	int label = 0;
	float prob = 0.0;
};

struct PreParam
{
	float ratio = 1.0f;
	float dw = 0.0f;
	float dh = 0.0f;
	float height = 0;
	float width = 0;
};

int num_labels = 1000;
float score_thres = 0.1f;
int topk = 3;

const std::vector<std::string> CLASS_NAMES = {
"tench","goldfish","great_white_shark","tiger_shark","hammerhead","electric_ray","stingray","cock","hen","ostrich","brambling","goldfinch","house_finch","junco","indigo_bunting","robin","bulbul","jay","magpie","chickadee","water_ouzel","kite","bald_eagle","vulture","great_grey_owl","European_fire_salamander","common_newt","eft","spotted_salamander","axolotl","bullfrog","tree_frog","tailed_frog","loggerhead","leatherback_turtle","mud_turtle","terrapin","box_turtle","banded_gecko","common_iguana","American_chameleon","whiptail","agama","frilled_lizard","alligator_lizard","Gila_monster","green_lizard","African_chameleon","Komodo_dragon","African_crocodile","American_alligator","triceratops","thunder_snake","ringneck_snake","hognose_snake","green_snake","king_snake","garter_snake","water_snake","vine_snake","night_snake","boa_constrictor","rock_python","Indian_cobra","green_mamba","sea_snake","horned_viper","diamondback","sidewinder","trilobite","harvestman","scorpion","black_and_gold_garden_spider","barn_spider","garden_spider","black_widow","tarantula","wolf_spider","tick","centipede","black_grouse","ptarmigan","ruffed_grouse","prairie_chicken","peacock","quail","partridge","African_grey","macaw","sulphur-crested_cockatoo","lorikeet","coucal","bee_eater","hornbill","hummingbird","jacamar","toucan","drake","red-breasted_merganser","goose","black_swan","tusker","echidna","platypus","wallaby","koala","wombat","jellyfish","sea_anemone","brain_coral","flatworm","nematode","conch","snail","slug","sea_slug","chiton","chambered_nautilus","Dungeness_crab","rock_crab","fiddler_crab","king_crab","American_lobster","spiny_lobster","crayfish","hermit_crab","isopod","white_stork","black_stork","spoonbill","flamingo","little_blue_heron","American_egret","bittern","crane_(bird)","limpkin","European_gallinule","American_coot","bustard","ruddy_turnstone","red-backed_sandpiper","redshank","dowitcher","oystercatcher","pelican","king_penguin","albatross","grey_whale","killer_whale","dugong","sea_lion","Chihuahua","Japanese_spaniel","Maltese_dog","Pekinese","Shih-Tzu","Blenheim_spaniel","papillon","toy_terrier","Rhodesian_ridgeback","Afghan_hound","basset","beagle","bloodhound","bluetick","black-and-tan_coonhound","Walker_hound","English_foxhound","redbone","borzoi","Irish_wolfhound","Italian_greyhound","whippet","Ibizan_hound","Norwegian_elkhound","otterhound","Saluki","Scottish_deerhound","Weimaraner","Staffordshire_bullterrier","American_Staffordshire_terrier","Bedlington_terrier","Border_terrier","Kerry_blue_terrier","Irish_terrier","Norfolk_terrier","Norwich_terrier","Yorkshire_terrier","wire-haired_fox_terrier","Lakeland_terrier","Sealyham_terrier","Airedale","cairn","Australian_terrier","Dandie_Dinmont","Boston_bull","miniature_schnauzer","giant_schnauzer","standard_schnauzer","Scotch_terrier","Tibetan_terrier","silky_terrier","soft-coated_wheaten_terrier","West_Highland_white_terrier","Lhasa","flat-coated_retriever","curly-coated_retriever","golden_retriever","Labrador_retriever","Chesapeake_Bay_retriever","German_short-haired_pointer","vizsla","English_setter","Irish_setter","Gordon_setter","Brittany_spaniel","clumber","English_springer","Welsh_springer_spaniel","cocker_spaniel","Sussex_spaniel","Irish_water_spaniel","kuvasz","schipperke","groenendael","malinois","briard","kelpie","komondor","Old_English_sheepdog","Shetland_sheepdog","collie","Border_collie","Bouvier_des_Flandres","Rottweiler","German_shepherd","Doberman","miniature_pinscher","Greater_Swiss_Mountain_dog","Bernese_mountain_dog","Appenzeller","EntleBucher","boxer","bull_mastiff","Tibetan_mastiff","French_bulldog","Great_Dane","Saint_Bernard","Eskimo_dog","malamute","Siberian_husky","dalmatian","affenpinscher","basenji","pug","Leonberg","Newfoundland","Great_Pyrenees","Samoyed","Pomeranian","chow","keeshond","Brabancon_griffon","Pembroke","Cardigan","toy_poodle","miniature_poodle","standard_poodle","Mexican_hairless","timber_wolf","white_wolf","red_wolf","coyote","dingo","dhole","African_hunting_dog","hyena","red_fox","kit_fox","Arctic_fox","grey_fox","tabby","tiger_cat","Persian_cat","Siamese_cat","Egyptian_cat","cougar","lynx","leopard","snow_leopard","jaguar","lion","tiger","cheetah","brown_bear","American_black_bear","ice_bear","sloth_bear","mongoose","meerkat","tiger_beetle","ladybug","ground_beetle","long-horned_beetle","leaf_beetle","dung_beetle","rhinoceros_beetle","weevil","fly","bee","ant","grasshopper","cricket","walking_stick","cockroach","mantis","cicada","leafhopper","lacewing","dragonfly","damselfly","admiral","ringlet","monarch","cabbage_butterfly","sulphur_butterfly","lycaenid","starfish","sea_urchin","sea_cucumber","wood_rabbit","hare","Angora","hamster","porcupine","fox_squirrel","marmot","beaver","guinea_pig","sorrel","zebra","hog","wild_boar","warthog","hippopotamus","ox","water_buffalo","bison","ram","bighorn","ibex","hartebeest","impala","gazelle","Arabian_camel","llama","weasel","mink","polecat","black-footed_ferret","otter","skunk","badger","armadillo","three-toed_sloth","orangutan","gorilla","chimpanzee","gibbon","siamang","guenon","patas","baboon","macaque","langur","colobus","proboscis_monkey","marmoset","capuchin","howler_monkey","titi","spider_monkey","squirrel_monkey","Madagascar_cat","indri","Indian_elephant","African_elephant","lesser_panda","giant_panda","barracouta","eel","coho","rock_beauty","anemone_fish","sturgeon","gar","lionfish","puffer","abacus","abaya","academic_gown","accordion","acoustic_guitar","aircraft_carrier","airliner","airship","altar","ambulance","amphibian","analog_clock","apiary","apron","ashcan","assault_rifle","backpack","bakery","balance_beam","balloon","ballpoint","Band_Aid","banjo","bannister","barbell","barber_chair","barbershop","barn","barometer","barrel","barrow","baseball","basketball","bassinet","bassoon","bathing_cap","bath_towel","bathtub","beach_wagon","beacon","beaker","bearskin","beer_bottle","beer_glass","bell_cote","bib","bicycle-built-for-two","bikini","binder","binoculars","birdhouse","boathouse","bobsled","bolo_tie","bonnet","bookcase","bookshop","bottlecap","bow","bow_tie","brass","brassiere","breakwater","breastplate","broom","bucket","buckle","bulletproof_vest","bullet_train","butcher_shop","cab","caldron","candle","cannon","canoe","can_opener","cardigan","car_mirror","carousel","carpenter's_kit","carton","car_wheel","cash_machine","cassette","cassette_player","castle","catamaran","CD_player","cello","cellular_telephone","chain","chainlink_fence","chain_mail","chain_saw","chest","chiffonier","chime","china_cabinet","Christmas_stocking","church","cinema","cleaver","cliff_dwelling","cloak","clog","cocktail_shaker","coffee_mug","coffeepot","coil","combination_lock","computer_keyboard","confectionery","container_ship","convertible","corkscrew","cornet","cowboy_boot","cowboy_hat","cradle","crane_(machine)","crash_helmet","crate","crib","Crock_Pot","croquet_ball","crutch","cuirass","dam","desk","desktop_computer","dial_telephone","diaper","digital_clock","digital_watch","dining_table","dishrag","dishwasher","disk_brake","dock","dogsled","dome","doormat","drilling_platform","drum","drumstick","dumbbell","Dutch_oven","electric_fan","electric_guitar","electric_locomotive","entertainment_center","envelope","espresso_maker","face_powder","feather_boa","file","fireboat","fire_engine","fire_screen","flagpole","flute","folding_chair","football_helmet","forklift","fountain","fountain_pen","four-poster","freight_car","French_horn","frying_pan","fur_coat","garbage_truck","gasmask","gas_pump","goblet","go-kart","golf_ball","golfcart","gondola","gong","gown","grand_piano","greenhouse","grille","grocery_store","guillotine","hair_slide","hair_spray","half_track","hammer","hamper","hand_blower","hand-held_computer","handkerchief","hard_disc","harmonica","harp","harvester","hatchet","holster","home_theater","honeycomb","hook","hoopskirt","horizontal_bar","horse_cart","hourglass","iPod","iron","jack-o'-lantern","jean","jeep","jersey","jigsaw_puzzle","jinrikisha","joystick","kimono","knee_pad","knot","lab_coat","ladle","lampshade","laptop","lawn_mower","lens_cap","letter_opener","library","lifeboat","lighter","limousine","liner","lipstick","Loafer","lotion","loudspeaker","loupe","lumbermill","magnetic_compass","mailbag","mailbox","maillot_(tights)","maillot_(tank_suit)","manhole_cover","maraca","marimba","mask","matchstick","maypole","maze","measuring_cup","medicine_chest","megalith","microphone","microwave","military_uniform","milk_can","minibus","miniskirt","minivan","missile","mitten","mixing_bowl","mobile_home","Model_T","modem","monastery","monitor","moped","mortar","mortarboard","mosque","mosquito_net","motor_scooter","mountain_bike","mountain_tent","mouse","mousetrap","moving_van","muzzle","nail","neck_brace","necklace","nipple","notebook","obelisk","oboe","ocarina","odometer","oil_filter","organ","oscilloscope","overskirt","oxcart","oxygen_mask","packet","paddle","paddlewheel","padlock","paintbrush","pajama","palace","panpipe","paper_towel","parachute","parallel_bars","park_bench","parking_meter","passenger_car","patio","pay-phone","pedestal","pencil_box","pencil_sharpener","perfume","Petri_dish","photocopier","pick","pickelhaube","picket_fence","pickup","pier","piggy_bank","pill_bottle","pillow","ping-pong_ball","pinwheel","pirate","pitcher","plane","planetarium","plastic_bag","plate_rack","plow","plunger","Polaroid_camera","pole","police_van","poncho","pool_table","pop_bottle","pot","potter's_wheel","power_drill","prayer_rug","printer","prison","projectile","projector","puck","punching_bag","purse","quill","quilt","racer","racket","radiator","radio","radio_telescope","rain_barrel","recreational_vehicle","reel","reflex_camera","refrigerator","remote_control","restaurant","revolver","rifle","rocking_chair","rotisserie","rubber_eraser","rugby_ball","rule","running_shoe","safe","safety_pin","saltshaker","sandal","sarong","sax","scabbard","scale","school_bus","schooner","scoreboard","screen","screw","screwdriver","seat_belt","sewing_machine","shield","shoe_shop","shoji","shopping_basket","shopping_cart","shovel","shower_cap","shower_curtain","ski","ski_mask","sleeping_bag","slide_rule","sliding_door","slot","snorkel","snowmobile","snowplow","soap_dispenser","soccer_ball","sock","solar_dish","sombrero","soup_bowl","space_bar","space_heater","space_shuttle","spatula","speedboat","spider_web","spindle","sports_car","spotlight","stage","steam_locomotive","steel_arch_bridge","steel_drum","stethoscope","stole","stone_wall","stopwatch","stove","strainer","streetcar","stretcher","studio_couch","stupa","submarine","suit","sundial","sunglass","sunglasses","sunscreen","suspension_bridge","swab","sweatshirt","swimming_trunks","swing","switch","syringe","table_lamp","tank","tape_player","teapot","teddy","television","tennis_ball","thatch","theater_curtain","thimble","thresher","throne","tile_roof","toaster","tobacco_shop","toilet_seat","torch","totem_pole","tow_truck","toyshop","tractor","trailer_truck","tray","trench_coat","tricycle","trimaran","tripod","triumphal_arch","trolleybus","trombone","tub","turnstile","typewriter_keyboard","umbrella","unicycle","upright","vacuum","vase","vault","velvet","vending_machine","vestment","viaduct","violin","volleyball","waffle_iron","wall_clock","wallet","wardrobe","warplane","washbasin","washer","water_bottle","water_jug","water_tower","whiskey_jug","whistle","wig","window_screen","window_shade","Windsor_tie","wine_bottle","wing","wok","wooden_spoon","wool","worm_fence","wreck","yawl","yurt","web_site","comic_book","crossword_puzzle","street_sign","traffic_light","book_jacket","menu","plate","guacamole","consomme","hot_pot","trifle","ice_cream","ice_lolly","French_loaf","bagel","pretzel","cheeseburger","hotdog","mashed_potato","head_cabbage","broccoli","cauliflower","zucchini","spaghetti_squash","acorn_squash","butternut_squash","cucumber","artichoke","bell_pepper","cardoon","mushroom","Granny_Smith","strawberry","orange","lemon","fig","pineapple","banana","jackfruit","custard_apple","pomegranate","hay","carbonara","chocolate_sauce","dough","meat_loaf","pizza","potpie","burrito","red_wine","espresso","cup","eggnog","alp","bubble","cliff","coral_reef","geyser","lakeside","promontory","sandbar","seashore","valley","volcano","ballplayer","groom","scuba_diver","rapeseed","daisy","yellow_lady's_slipper","corn","acorn","hip","buckeye","coral_fungus","agaric","gyromitra","stinkhorn","earthstar","hen-of-the-woods","bolete","ear","toilet_tissue"
};

const std::vector<std::vector<unsigned int>> COLORS = {
    {0, 114, 189},   {217, 83, 25},   {237, 177, 32},  {126, 47, 142},  {119, 172, 48},  {77, 190, 238},
    {162, 20, 47},   {76, 76, 76},    {153, 153, 153}, {255, 0, 0},     {255, 128, 0},   {191, 191, 0},
    {0, 255, 0},     {0, 0, 255},     {170, 0, 255},   {85, 85, 0},     {85, 170, 0},    {85, 255, 0},
    {170, 85, 0},    {170, 170, 0},   {170, 255, 0},   {255, 85, 0},    {255, 170, 0},   {255, 128, 0},
    {0, 85, 128},    {0, 170, 128},   {0, 255, 128},   {85, 0, 128},    {85, 85, 128},   {85, 170, 128},
    {85, 255, 128},  {170, 0, 128},   {170, 85, 128},  {170, 170, 128}, {170, 255, 128}, {255, 0, 128},
    {255, 85, 128},  {255, 170, 128}, {255, 255, 128}, {0, 85, 255},    {0, 170, 255},   {0, 255, 255},
    {85, 0, 255},    {85, 85, 255},   {85, 170, 255},  {85, 255, 255},  {170, 0, 255},   {170, 85, 255},
    {170, 170, 255}, {170, 255, 255}, {255, 0, 255},   {255, 85, 255},  {255, 170, 255}, {85, 0, 0},
    {128, 0, 0},     {170, 0, 0},     {212, 0, 0},     {255, 0, 0},     {0, 43, 0},      {0, 85, 0},
    {0, 128, 0},     {0, 170, 0},     {0, 212, 0},     {0, 255, 0},     {0, 0, 43},      {0, 0, 85},
    {0, 0, 128},     {0, 0, 170},     {0, 0, 212},     {0, 0, 255},     {0, 0, 0},       {36, 36, 36},
    {73, 73, 73},    {109, 109, 109}, {146, 146, 146}, {182, 182, 182}, {219, 219, 219}, {0, 114, 189},
    {80, 183, 189},  {128, 128, 0}
};

void postprocess(std::vector<Object>& objs, float* host_ptrs[1], PreParam pparam, int index)
{
	objs.clear();
    
    std::vector<float>    scores;
    std::vector<int>      labels;
    
    auto& dw = pparam.dw;
	auto& dh = pparam.dh;
	auto& width = pparam.width;
	auto& height = pparam.height;
	auto& ratio = pparam.ratio;

    auto* output_data = static_cast<float*>(host_ptrs[0]);
    int offset = index * num_labels;
    
    float* scores_ptr = output_data + offset;


    // Find top-k values and indices
    size_t k = std::min<size_t>(topk, num_labels); // Ensure k is within bounds
    std::vector<size_t> indices(num_labels);
    std::iota(indices.begin(), indices.end(), 0); // Fill indices with 0, 1, 2, ..., num_labels-1

    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(), 
                      [&](size_t i, size_t j) { return scores_ptr[i] > scores_ptr[j]; });

    for (size_t i = 0; i < k; ++i) {
        // std::cout << scores_ptr[indices[i]] << " (Index: " << indices[i] << ") ";
        Object obj;
        obj.prob  = scores_ptr[indices[i]];
        obj.label = indices[i];
        objs.push_back(obj);
        
    }
    std::cout << std::endl;
    
}

void draw_objects(
	const cv::Mat& image,
	cv::Mat& res,
	const std::vector<Object>& objs,
	const std::vector<std::string>& CLASS_NAMES,
	const std::vector<std::vector<unsigned int>>& COLORS
)
{
	res = image.clone();
	for (int i = 0; i < objs.size(); i++)
	{	
        Object obj = objs[i];
		cv::Scalar color = cv::Scalar(
			COLORS[obj.label % 80][0],
			COLORS[obj.label % 80][1],
			COLORS[obj.label % 80][2]
		);
        
		char text[256];
		sprintf(
			text,
			"%s %.1f%%",
			CLASS_NAMES[obj.label].c_str(),
			obj.prob * 100
		);

		int baseLine = 0;
		cv::Size label_size = cv::getTextSize(
			text,
			cv::FONT_HERSHEY_SIMPLEX,
			0.4,
			1,
			&baseLine
		);

		int x = (int)5;
		int y = (int)5 + i * (label_size.height+3);

		if (y > res.rows)
			y = res.rows;
    

		cv::rectangle(
			res,
			cv::Rect(x, y, label_size.width, label_size.height + baseLine),
			color,
			-1
		);

		cv::putText(
			res,
			text,
			cv::Point(x, y + label_size.height),
			cv::FONT_HERSHEY_SIMPLEX,
			0.4,
			{ 255, 255, 255 },
			1
		);
	}
}



#define MAX_IMAGE_INPUT_SIZE_THRESH 3000 * 3000

std::shared_ptr<spdlog::logger> logger = spdlog::stdout_color_mt("yolov8-classify");


class my_formatter_flag : public spdlog::custom_flag_formatter
{
public:
    void format(const spdlog::details::log_msg &msg, const std::tm &, spdlog::memory_buf_t &dest) override
    {
		char yi_log_level_text[9] = {0};
		if (msg.level == spdlog::level::level_enum::trace) {
		strcpy(yi_log_level_text, "TRACE");
		} else if (msg.level == spdlog::level::level_enum::debug) {
		strcpy(yi_log_level_text, "DEBUG");
		} else if (msg.level == spdlog::level::level_enum::warn) {
		strcpy(yi_log_level_text, "WARNING");
		} else if (msg.level == spdlog::level::level_enum::err) {
		strcpy(yi_log_level_text, "ERROR");
		} else if (msg.level == spdlog::level::level_enum::critical) {
		strcpy(yi_log_level_text, "CRITICAL");
		} else {
		strcpy(yi_log_level_text, "INFO");
		}
		dest.append(yi_log_level_text,
					yi_log_level_text + strlen(yi_log_level_text));
    }

    std::unique_ptr<custom_flag_formatter> clone() const override
    {
        return spdlog::details::make_unique<my_formatter_flag>();
    }
};


int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names) {
    DIR *p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        return -1;
    }

    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr) {
        if (strcmp(p_file->d_name, ".") != 0 &&
            strcmp(p_file->d_name, "..") != 0) {
            // Construct full path of the file
            std::string full_path = std::string(p_dir_name) + "/" + std::string(p_file->d_name);
            
            // Check if the file is a regular file
            struct stat file_stat;
            if (stat(full_path.c_str(), &file_stat) == 0 && S_ISREG(file_stat.st_mode)) {
                // Add only if it's a regular file
                file_names.push_back(p_file->d_name);
            }
        }
    }

    closedir(p_dir);
    return 0;
}

int main(int argc, char *argv[]){

    std::string jsonlogpattern = { "%^{\"timestamp\": \"%Y-%m-%dT%H:%M:%S.%eZ\",  \"logLevel\": \"%*\", \"logFacility\": null,  \"function\": \"%!\", \"file\": \"%s\", \"lineNo\": %#, \"message\": \"%v\"}%$" };
	
	auto formatter = std::make_unique<spdlog::pattern_formatter>();
    formatter->add_flag<my_formatter_flag>('*').set_pattern(jsonlogpattern);
    spdlog::set_formatter(std::move(formatter));


    uint8_t** hostbuffers;
    uint8_t** devicebuffers;

    float* crop_hostbuffer_debug = nullptr;

    int BatchSize = 2;

    // prepare host cache
    hostbuffers = (uint8_t**)malloc(sizeof(uint8_t*) * BatchSize);
    for(int i = 0 ; i < BatchSize; i++)
        CUDA_CHECK(cudaMallocHost(&hostbuffers[i], MAX_IMAGE_INPUT_SIZE_THRESH * 3)); // RGB

    // prepare device cache for input image:  All input RGB data
    devicebuffers = (uint8_t**)malloc(sizeof(uint8_t*) * BatchSize);
    for(int i = 0 ; i < BatchSize; i++)
        CUDA_CHECK(cudaMalloc(&devicebuffers[i], MAX_IMAGE_INPUT_SIZE_THRESH * 3)); // RGB


    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    int InputH = 224;
    int InputW = 224;
    //Load the backbone
    Logger g_logger;
    int min_NCHW_[4] = {BatchSize,3,InputH,InputW};
    int opt_NCHW_[4] = {BatchSize,3,InputH,InputW};
    int max_NCHW_[4] = {BatchSize,3,InputH,InputW};
    
    int* min_NCHW= &min_NCHW_[0];
    int* opt_NCHW= &opt_NCHW_[0];
    int* max_NCHW= &max_NCHW_[0];

    char* format = "NCHW";
    vector<const char*> INPUT_BLOB_NAME = {"images"};
    vector<const char*> OUTPUT_BLOB_NAME = {"output0"};

    SrcNetwork classify_v8(g_logger, format, min_NCHW, opt_NCHW, max_NCHW, INPUT_BLOB_NAME, OUTPUT_BLOB_NAME);

    std::string onnx_path = argv[1];

    // if trt_path doesn't exist then build the engine
    std::string onnx_path_str(onnx_path);
    SPDLOG_LOGGER_INFO(logger,"Onnx Path: {}", onnx_path_str);
    std::string trt_path = onnx_path_str.substr(0, onnx_path_str.find_last_of('.')) + "_batchsize_" +std::to_string(BatchSize)+".trt";
    SPDLOG_LOGGER_INFO(logger,"TRT Path: {}", trt_path);

    std::ifstream trt_file(trt_path, std::ios::binary);
    IExecutionContext* detector_context;
    if (!trt_file.good()) {
        classify_v8.buildEngine(strdup(onnx_path.c_str()), strdup(trt_path.c_str()));
        detector_context = classify_v8.getEngineContext(strdup(trt_path.c_str()), stream);
    }else{
        detector_context = classify_v8.getEngineContext(strdup(trt_path.c_str()), stream);
    }

    float* backbone_buffers[classify_v8.num_inputs + classify_v8.num_outputs];
    float* output_buffers[BatchSize * classify_v8.num_outputs];

    CUDA_CHECK(cudaMalloc((void**)&backbone_buffers[0], max_NCHW_[0] * max_NCHW_[1] * max_NCHW_[2] * max_NCHW_[3]  * sizeof(float)));    
    int i = 0;
    for (auto& bindings : classify_v8.output_bindings){
        
        size_t size = bindings.size * bindings.dsize;

        CUDA_CHECK(cudaMalloc((void**)&backbone_buffers[1 + i], max_NCHW_[0] * size  *  sizeof(float)));
        CUDA_CHECK(cudaMallocHost((void**)&output_buffers[i], max_NCHW_[0] * size  *  sizeof(float)));

        i += 1;
    }

    std::string img_dir = argv[2];
    std::vector<std::string> file_names;
    if (read_files_in_dir(img_dir.c_str(), file_names) < 0) {
        std::cerr << "read_files_in_dir failed." << std::endl;
        return -1;
    }

    for (size_t i = 0; i < file_names.size(); i += BatchSize) {
        
        std::vector<cv::Mat> img_batch;
        std::vector<std::string> img_name_batch;

        for (size_t j = i; j < i + BatchSize && j < file_names.size(); j++) {
            cv::Mat img = cv::imread(img_dir + "/" + file_names[j]);
            img_batch.push_back(img);
            img_name_batch.push_back(file_names[j]);
        }
        
        int currentBatchSize = img_batch.size();
        SPDLOG_LOGGER_INFO(logger,"Current Batch Size: {}", currentBatchSize); 
        
        //PREPROCESS
        float *buffer_idx = (float *)backbone_buffers[0];
        std::vector<PreParam> pparam_per_batch;
        for(int index=0; index < currentBatchSize; index++){

            cv::Mat img = img_batch[index].clone();

            if(img.empty()){
                SPDLOG_LOGGER_ERROR(logger, "Image {} is empty", img_name_batch[index]);
                continue;
            }

            if(img.channels() != 3){
                SPDLOG_LOGGER_ERROR(logger, "Image {} channels is not 3", img_name_batch[index]);
                continue;
            }

            if(img.rows * img.cols > MAX_IMAGE_INPUT_SIZE_THRESH ){
                SPDLOG_LOGGER_ERROR(logger, "Image {} size is too large", img_name_batch[index]);
                continue;
            }

            
            int rows = img.rows;
            int cols = img.cols;
                    
            const float inp_h = InputH;
            const float inp_w = InputW;
            float height = rows;
            float width = cols;

            float r = std::min(inp_h / height, inp_w / width);
            int padw = std::round(width * r);
            int padh = std::round(height * r);
            float dw = inp_w - padw;
            float dh = inp_h - padh;
            dw /= 2.0f;
            dh /= 2.0f;

            PreParam pparam;
            pparam.ratio = 1 / r;
            pparam.dw = dw;
            pparam.dh = dh;
            pparam.height = height;
            pparam.width = width;
            pparam_per_batch.push_back(pparam);

            size_t size_image = rows * cols * 3;
            hostbuffers[index] = img.data;
            CUDA_CHECK(cudaMemcpyAsync(devicebuffers[index], hostbuffers[index], size_image, cudaMemcpyHostToDevice, stream));
            
            float Imean_values[3]  = {0.0f, 0.0f, 0.0f};
            float Iscale_values[3] = {1.0f, 1.0f, 1.0f};
            int mid = min(height, width);
            int top = (height - mid) / 2;
            int left = (width - mid) / 2;
            cv::Rect context_crop(left, top, mid, mid);
            
            crop_resize_kernel_img(
                devicebuffers[index], cols, rows,        //src
                buffer_idx, InputW, InputH, //dst
                context_crop,      //  crop,
                &Imean_values[0],  //  Imean_values,
                &Iscale_values[0], //  Iscale_values,
                1,  //  letterbox,
                0,  //  scale_given,
                0,  //  size,
                1,  //  is_norm: Divide by 255 
                stream
            );
            buffer_idx += InputH * InputW * 3 ;        
        }
        
        SPDLOG_LOGGER_INFO(logger, "Calling inference");
        //INFERENCE
        detector_context->enqueue(currentBatchSize, (void**)backbone_buffers, stream, nullptr);
        SPDLOG_LOGGER_INFO(logger, "Calling inference done");
        
        //COPY RESULTS
        for (int i = 0; i < classify_v8.output_bindings.size(); i++)
        {
            size_t osize = classify_v8.output_bindings[i].size * classify_v8.output_bindings[i].dsize;
            CUDA_CHECK(cudaMemcpyAsync(output_buffers[i],
                backbone_buffers[i + 1],
                osize,
                cudaMemcpyDeviceToHost,
                stream)
            );

        }

        cudaStreamSynchronize(stream);
        
        SPDLOG_LOGGER_INFO(logger, "Calling postprocess");
        
        
        for(int index=0; index < currentBatchSize; index++){
            std::vector<Object> objs;
            postprocess(objs, output_buffers, pparam_per_batch[index], index);
            
            cv::Mat res;
            draw_objects(img_batch[index], res, objs, CLASS_NAMES, COLORS);

            // Save image
            cv::imwrite("/app/results/v8classify_" + img_name_batch[index], res);
            
        }
        SPDLOG_LOGGER_INFO(logger, "Calling postprocess done");
        
        
    }
    

    return 0;
}
