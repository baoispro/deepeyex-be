package enums

type Specialty string

const (
// 🔹 Nhóm chuyên khoa về mắt
	SpecialtyOphthalmology      Specialty = "ophthalmology"       // Nhãn khoa tổng quát
	SpecialtyRetina             Specialty = "retina"              // Chuyên khoa võng mạc
	SpecialtyCornea             Specialty = "cornea"              // Chuyên khoa giác mạc
	SpecialtyGlaucoma           Specialty = "glaucoma"            // Cườm nước
	SpecialtyRefractiveSurgery  Specialty = "refractive_surgery"  // Phẫu thuật khúc xạ (LASIK)
	SpecialtyPediatricOphthalm  Specialty = "pediatric_ophthalm"  // Nhãn khoa nhi

	// 🔹 Nhóm quản lý & hỗ trợ (nếu hệ thống có nhân viên khác)
	SpecialtyScheduleManager    Specialty = "schedule_manager"    // Quản lý lịch bác sĩ
)
