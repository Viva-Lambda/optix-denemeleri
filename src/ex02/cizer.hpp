#ifndef CIZER_HPP
#define CIZER_HPP
#include "baslatma.hpp"
#include "cudabellek.hpp"
#include <optix_function_table_definition.h>
#include <vector>

extern "C" char koyulan_ptx_kodu[];

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) IsinyaratKaydi {
  __align__(
      OPTIX_SBT_RECORD_ALIGNMENT) char baslik[OPTIX_SBT_RECORD_HEADER_SIZE]
      // simdilik bos ileriki derslerde içini dolduracagiz
      void *data;
};
struct __align__(OPTIX_SBT_RECOR_ALIGNMENT) IskalamaKaydi {
  __align__(
      OPTIX_SBT_RECORD_ALIGNMENT) char baslik[OPTIX_SBT_RECORD_HEADER_SIZE]
      // simdilik bos ileriki derslerde içini dolduracagiz
      void *data;
};
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) VurulanlarKaydi {
  __align__(
      OPTIX_SBT_RECORD_ALIGNMENT) char baslik[OPTIX_SBT_RECORD_HEADER_SIZE]
      // simdilik bos ileriki derslerde içini dolduracagiz
      int vurulanID;
};

class OrnekCizer {
public:
  OrnekCizer(); // kurucu
  void boyutlandir(const glm::ivec2 &cerceveBoyutu);
  void ciz();
  void pikseliIndir(std::vector<uint32_t> renkler);

protected:
  /** Optix'i baslatir ve hata kontrollerini yapar*/
  void optixBaslat(); // initOptix();

  /** Optix'in çalisacagi aygiti yaratir ve gerekli ayarlamalari yapar */
  void baglamYarat(); // createContext();

  /** CUDA çalistiran modulu yaratir*/
  void modulYarat(); // createModule();

  /** isin yaratma programini (raygenProgram) olusturur*/
  void isinYaratmaProgramiYarat(); // createRaygenProgram();

  /** iskalama programini (missProgram) olusturur */
  void iskalamaProgramiYarat(); // createMissProgram();

  /** vurulanlar (hitGroup) programini olusturur */
  void vurulanlarProgramiYarat(); // createHitGroupProgram();

  /** program veri hattini (pipeline) olusturur */
  void veriHattiYarat(); // createPipeline();

  /** golgeleyici baglama tablosunu (Shader Binding Table) yaratir */
  void golgeleyiciBaglamaTablosuOlustur(); // buildSBT();

protected:
  /** Cizicimizin çizim surecinde kullanacagi aracilar */

  /** @{ Optix API'nin uzerinde çalisacagi veri akisi hatti ve cuda aygiti
   * ve cuda baglami */
  CUcontext cudaBaglam;
  CUstream cudaAkisi;
  cudaDeviceProp aygitOzellikleri;

  /** @} */

  /** Optix aygit baglami */
  OptixDeviceContext optixBaglami;

  /** @{ OptiX veri hatti: Temelde bunu olusturmayi ogreniyoruz. */

  OptixPipeline optixVeriHatti;
  OptixPipelineCompileOptions opxVeriHattiDerlemeSecenekleri;
  OptixPipelineLinkOptions opxVeriHattiLinkSecenekleri

      /** @} */

      /** @{ Ilgili programi yoneten birim */
      OptixModule opxBirimi;
  OptixModuleCompileOptions opxBirimSecenekleri;

  /** @} */

  /** @{ Kullanacagimiz program gruplari ve onlarin etrafinda sekillenen
   * diger objeler,  golgeleyici baglama tablosu vb */

  // Gruplar
  std::vector<OptixProgramGroup> isinYaratmaPGlari;
  std::vector<OptixProgramGroup> iskalamaPGlari;
  std::vector<OptixProgramGroup> vurulanlarPGlari;

  // Gruplarin saklanacagi arabellekler
  CUDABuffer IsinyaratKaydiAraBellegi;
  CUDABuffer IskalamaKaydiAraBellegi;
  CUDABuffer VurulanlarKaydiAraBellegi;

  // Golgeleyici Baglama Tablosu
  OptixShaderBindingTable gbt = {};

  /** @} */

  /** @{ Cizimi baslatma parametrelerimiz.*/

  BaslatmaParametreleri baslatmaPrmleri;
  CUDABuffer baslatmaPrmleriArabellegi;

  /** @} */

  CUDABuffer renkArabellegi;
};

OrnekCizer::OrnekCizer() {

  optixBaslat(); // baslatma kodu

  std::cout << "Optix baglami olusturuluyor ..." << std::endl;
  baglamYarat();

  std::cout << "Modulu kurulumuna baslaniyor ..." << std::endl;
  modulYarat();
  std::cout << "Modulu kurulumu tamamlandi ..." << std::endl;

  std::cout << "Isin yaratma programi olusturuluyor ..." << std::endl;
  isinYaratmaProgramiYarat();
  std::cout << "Isin yaratma programi olusturuldu ..." << std::endl;

  std::cout << "Iskalama programi olusturuluyor ..." << std::endl;
  iskalamaProgramiYarat();
  std::cout << "Iskalama programi olusturuldu ..." << std::endl;

  std::cout << "Vurulanlar programi olusturuluyor ..." << std::endl;
  vurulanlarProgramiYarat();
  std::cout << "Vurulanlar programi olusturuldu ..." << std::endl;

  std::cout << "Veri hatti olusturuluyor ..." << std::endl;
  veriHattiYarat();
  std::cout << "Veri hatti olusturuldu ..." << std::endl;

  std::cout << "Golgeleyici Baglama Tablosu (GBT) olusturuluyor" << std::endl;
  golgeleyiciBaglamaTablosuOlustur();

  // baslatma parametrelerini ayarla
  BaslatmaParametreleri.alloc(sizeof(BaslatmaParametreleri));

  std::cout << "Optix Cizerin kurulumu tamamlandi" << std::endl;
}

void OrnekCizer::optixBaslat() {

  // 1. Optix uyumlu cihaz kontrolu
  cudaFree(0);
  int cihazSayisi;

  // cuda ile uyumlu cihazi aratir ve yarattigimiz degiskene sayisini koyar
  cudaGetDeviceCount(&cihazSayisi);

  // eger koydugu sayi 0 ise CUDA ile uyumlu cihaz bulamamis demektir
  if (cihazSayisi == 0) {

    throw std::runtime_error("CUDA ile uyumlu cihaz bulunamamistir");

    // CUDA ile uyumlu cihaz yoksa optix'i kullanamayiz demektir
    // dolayisiyla bu hatayi yakalamanin bir anlami yok. Program burada
    // durmali
  }
  OptixResult optixiBaslat = optixInit(); // optixInit optix sdk'nin bir parçasi
  OPTIX_KONTROL(optixiBaslat);
}

static void baglam_log(unsigned int seviye, const char *etiket,
                       const char *mesaj, void *) {
  std::cerr << std::to_string(seviye) << etiket << mesaj << std::endl;
}

void OrnekCizer::baglamYarat() {

  // simdilik her seyi tek bir aygitta, ana aygitta yapalim
  const int aygitID = 0;
  CUDA_KONTROL(cudaSetDevice(aygitID));

  // yeni asenkronize veri akis yolu olustur
  CUDA_KONTROL(cudaStreamCreate(&cudaAkisi));

  // aygit ozellikleri al
  cudaGetDeviceProperties(&aygitOzellikleri, aygitID);
  std::cout << "calistirilan aygit: " << aygitOzellikleri.name << std::endl;

  // baglami olustur
  CUcontext cudaBaglam CUresult cudaSonucu = cuCtxGetCurrent(&cudaBaglam);
  if (cudaSonucu != CUDA_SUCCESS) {
    std::stringstream uyari;
    uyari << "Mevcut Cuda Baglami alinamadi. Hata kodu: " << cudaSonucu;
    throw std::runtime_error(uyari.str());
  }

  OptixDeviceContext optixBaglami;
  OPTIX_KONTROL(optixDeviceContextCreate(cudaBaglam, 0, &optixBaglami));
  OPTIX_KONTROL(
      optixDeviceContextSetLogCallback(optixBaglami, baglam_log, nullptr, 4));
}

OrnekCizer::modulYarat() {
  opxBirimSecenekleri.maxRegisterCount = 50;
  opxBirimSecenekleri.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
  opxBirimSecenekleri.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

  opxVeriHattiDerlemeSecenekleri = {};
  opxVeriHattiDerlemeSecenekleri.traversableGraphFlags =
      OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
  opxVeriHattiDerlemeSecenekleri.usesMotionBlur = false;
  opxVeriHattiDerlemeSecenekleri.numPayloadValues = 2;
  opxVeriHattiDerlemeSecenekleri.numAttributeValues = 2;
  opxVeriHattiDerlemeSecenekleri.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
  opxVeriHattiDerlemeSecenekleri.pipelineLaunchParamsVariableName =
      "optixLaunchParams";
  opxVeriHattiLinkSecenekleri.overrideUsesMotionBlur = false;
  opxVeriHattiLinkSecenekleri.maxTraceDepth = 2;

  const std::string ptxKodu = koyulan_ptx_kodu;

  char mlog[2048];
  size_t sizeof_mlog = sizeof(mlog);
  OPTIX_KONTROL(
          optixModuleCreateFromPTX(
              optixBaglami,
              &opxBirimSecenekleri,
              &opxVeriHattiDerlemeSecenekleri,
              ptxKodu.c_str(),
              ptxKodu.size(),
              mlog, &sizeof_mlog,
              &opxBirimi
          );
  if(sizeof_mlog > 1){
    std::cout << "module log: " << mlog << std::endl;
  }
}

OrnekCizer::isinYaratmaProgramiYarat() {
  // tek isinli program
  isinYaratmaPGlari.resize(1);

  OptixProgramGroupOptions pgSecenek = {};
  OptixProgramGroupDesc pgTarif = {};
  pgTarif.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  pgTarif.raygen.module = opxBirimi;
  pgTarif.raygen.entryFunctionName = "__raygen_renderFrame";

  //
  char raygenLog[2048];
  size_t sizeof_rLog = sizeof(raygenLog);
  OPTIX_KONTROL(optixProgramGroupCreate(optixBaglami, &pgTarif, 1, &pgSecenek,
                                        raygenLog, sizeof_rLog,
                                        &isinYaratmaPGlari[0]));
  if (sizeof_rLog > 1) {
    std::cout << "isin yaratma logu: " << raygenLog << std::endl;
  }
}
OrnekCizer::iskalamaProgramiYarat() {
  iskalamaPGlari.resize(1);
  OptixProgramGroupOptions pgSecenek = {};
  OptixProgramGroupDesc pgTarif = {};
  pgTarif.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
  pgTarif.miss.module = opxBirimi;
  pgTarif.miss.entryFunctionName = "__miss__radiance";

  //
  char missLog[2048];
  size_t sizeof_mLog = sizeof(missLog);
  OPTIX_KONTROL(optixProgramGroupCreate(optixBaglami, &pgTarif, 1, &pgSecenek,
                                        missLog, sizeof_mLog,
                                        &iskalamaPGlari[0]));
  if (sizeof_mLog > 1) {
    std::cout << "miss logu: " << missLog << std::endl;
  }
}
OrnekCizer::vurulanlarProgramiYarat() {
  vurulanlarPGlari.resize(1);
  OptixProgramGroupOptions pgSecenek = {};
  OptixProgramGroupDesc pgTarif = {};
  pgTarif.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  pgTarif.hitgroup.moduleCH = opxBirimi;
  pgTarif.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
  pgTarif.hitgroup.moduleAH = opxBirimi;
  pgTarif.hitgroup.entryFunctionNameAH = "__anyhit__radiance";

  //
  char vlog[2048];
  size_t sizeof_vLog = sizeof(vlog);
  OPTIX_KONTROL(optixProgramGroupCreate(optixBaglami, &pgTarif, 1, &pgSecenek,
                                        vlog, sizeof_vLog,
                                        &vurulanlarPGlari[0]));
  if (sizeof_vLog > 1) {
    std::cout << "vurulanlar logu: " << vlog << std::endl;
  }
}
OrnekCizer::veriHattiYarat() {
  // program akisini belirler
  std::vector<OptixProgramGroup> programGruplari;
  for (auto pg : isinYaratmaPGlari) {
    programGruplari.push_back(pg);
  }
  for (auto pg : iskalamaPGlari) {
    programGruplari.push_back(pg);
  }
  for (auto pg : vurulanlarPGlari) {
    programGruplari.push_back(pg);
  }
  char log[2048];
  size_t slog = sizeof(log);
  OPTIX_KONTROL(optixPipelineCreate(
      optixBaglami, &opxVeriHattiDerlemeSecenekleri,
      &opxVeriHattiLinkSecenekleri, programGruplari.data(),
      static_cast<int>(programGruplari.size()), log, slog, &optixVeriHatti));

  if (slog > 1) {
    std::cout "pipeline logu: " << log << std::endl;
  }
  OPTIX_KONTROL(optixPipelineSetStackSize(
      optixVeriHatti, // yigin boyutunun kendisine gore
      // ayarlanacagi veri hatti
      2 * 1024, // any hit programlari tarafindan
      // çagrilacaklar için gerekli dogrudan yigin
      // boyutu
      2 * 1024, // isin yaratma, iskalama, en yakin çarpma
      // için gerekli dogrudan yigin boyutu
      2 * 1024, // surdurmek için gereken yigin boyutu
      1         // isleme konulacak sahne grafiginde katedilecek
      // derinligin boyutu
      ));
  if (slog > 1) {
    std::cout "pipeline logu: " << log << std::endl;
  }
}
OrnekCizer::golgeleyiciBaglamaTablosuOlustur() {
  // -------------------
  // isin yaratma programi olustur
  // -------------------
  std::vector<IsinyaratKaydi> isinYaratKayitlari;

  for (int i = 0; i < isinYaratmaPGlari.size(); i++) {
    IsinyaratKaydi kayit;
    OPTIX_KONTROL(optixSbtRecordPackHeader(isinYaratmaPGlari[i], &kayit));
    kayit.data = nullptr;
    isinYaratKayitlari.push_back(kayit);
  }
  IsinyaratKaydiAraBellegi.tahsis_et_yukle(isinYaratKayitlari);
  gbt.raygenRecord = IsinyaratKaydiAraBellegi.aygit_imi();

  // -------------------
  // isklama programi olustur
  // -------------------
  std::vector<IskalamaKaydi> isklamaKayitlari;

  for (int i = 0; i < iskalamaPGlari.size(); i++) {
    IskalamaKaydi kayit;
    OPTIX_KONTROL(optixSbtRecordPackHeader(iskalamaPGlari[i], &kayit));
    kayit.data = nullptr;
    isklamaKayitlari.push_back(kayit);
  }
  IskalamaKaydiAraBellegi.tahsis_et_yukle(isklamaKayitlari);
  gbt.missRecordBase = IskalamaKaydiAraBellegi.aygit_imi();
  gbt.missRecordStrideInBytes = sizeof(IskalamaKaydi);
  gbt.missRecordCount = static_cast<int> isklamaKayitlari.size();
  // -------------------
  // vurma programi olustur
  // -------------------
  int nbobj = 1;
  std::vector<VurulanlarKaydi> vurulanKayitlari;
  for (int i = 0; i < nbobj; i++) {
    VurulanlarKaydi kayit;
    OPTIX_KONTROL(optixSbtRecordPackHeader(vurulanlarPGlari[i], &kayit));
    kayit.vurulanID = i;
    vurulanKayitlari.push_back(kayit);
  }
  VurulanlarKaydiAraBellegi.tahsis_et_yukle(vurulanKayitlari);
  gbt.hitgroupRecordBase = VurulanlarKaydiAraBellegi.aygit_imi();
  gbt.hitgroupRecordStrideInBytes = sizeof(VurulanlarKaydi);
  gbt.hitgroupRecordCount = static_cast<int> vurulanKayitlari.size();
}

#endif
