package appointmentservice

import (
	"errors"
	"fmt"
	"hospital-service/internal/models/appointment"
	"hospital-service/internal/models/doctor"
	"hospital-service/internal/repositories/appointmentrepo"
	"hospital-service/internal/repositories/doctorrepo"
	"log"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/xuri/excelize/v2"
)

type Doctor = doctor.Doctor

type TimeSlotService struct {
	timeSlotRepo *appointmentrepo.TimeSlotRepo
	doctorRepo   *doctorrepo.DoctorRepo
	appointmentRepo  *appointmentrepo.AppointmentRepo

}

func NewTimeSlotService(timeSlotRepo *appointmentrepo.TimeSlotRepo, doctorRepo *doctorrepo.DoctorRepo, appointmentRepo *appointmentrepo.AppointmentRepo) *TimeSlotService {
	return &TimeSlotService{
		timeSlotRepo: timeSlotRepo,
		doctorRepo:   doctorRepo,
		appointmentRepo: appointmentRepo,

	}
}

type TimeSlotInput struct {
	StartTime time.Time
	EndTime   time.Time
	Capacity  int
}

type CreateMultiShiftSlotsRequest struct {
    DoctorID string            `json:"doctor_id" binding:"required"`
    Shifts   []ShiftSelection  `json:"shifts" binding:"required"`
}

type ShiftSelection struct {
    Date  string   `json:"date" binding:"required"`  // YYYY-MM-DD
    Slots []string `json:"slots" binding:"required"` // ["morning","afternoon","evening"]
}


//
// ---------------- CRUD ----------------
//

// Create tạo 1 time slot thủ công
func (s *TimeSlotService) Create(doctorID string, startTime, endTime time.Time, capacity int) (*appointment.TimeSlot, error) {
	if doctorID == "" {
		return nil, errors.New("doctor ID is required")
	}
	if capacity <= 0 {
		return nil, errors.New("capacity must be greater than 0")
	}
	if !startTime.Before(endTime) {
		return nil, errors.New("startTime must be before endTime")
	}

	// Kiểm tra trùng
	count, err := s.timeSlotRepo.CountOverlapping(doctorID, startTime, endTime)
	if err != nil {
		return nil, fmt.Errorf("failed to check overlapping slots: %v", err)
	}
	if count > 0 {
		return nil, errors.New("doctor already has a timeslot in this time range")
	}

	slot := &appointment.TimeSlot{
		SlotID:    generateSlotID(),
		DoctorID:  doctorID,
		StartTime: startTime,
		EndTime:   endTime,
		Capacity:  capacity,
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}

	if err := s.timeSlotRepo.Create(slot); err != nil {
		return nil, err
	}
	return slot, nil
}

func (s *TimeSlotService) GetByID(id string) (*appointment.TimeSlot, error) {
	return s.timeSlotRepo.FindByID(id)
}

func (s *TimeSlotService) GetByDoctorID(doctorID string) ([]appointment.TimeSlot, error) {
	return s.timeSlotRepo.FindByDoctorID(doctorID)
}

func (s *TimeSlotService) Update(slotID string, startTime, endTime *time.Time, capacity *int) (*appointment.TimeSlot, error) {
	slot, err := s.timeSlotRepo.FindByID(slotID)
	if err != nil {
		return nil, fmt.Errorf("timeslot not found: %v", err)
	}

	if startTime != nil {
		slot.StartTime = *startTime
	}
	if endTime != nil {
		slot.EndTime = *endTime
	}
	if capacity != nil {
		if *capacity <= 0 {
			return nil, errors.New("capacity must be greater than 0")
		}
		slot.Capacity = *capacity
	}

	if !slot.StartTime.Before(slot.EndTime) {
		return nil, errors.New("startTime must be before endTime")
	}

	count, err := s.timeSlotRepo.CountOverlapping(slot.DoctorID, slot.StartTime, slot.EndTime)
	if err != nil {
		return nil, fmt.Errorf("failed to check overlapping slots: %v", err)
	}
	if count > 0 {
		return nil, errors.New("doctor already has a timeslot in this time range")
	}

	slot.UpdatedAt = time.Now()
	if err := s.timeSlotRepo.Update(slot); err != nil {
		return nil, err
	}

	return slot, nil
}

func (s *TimeSlotService) Delete(id string) error {
	return s.timeSlotRepo.Delete(id)
}

func (s *TimeSlotService) ListAll() ([]appointment.TimeSlot, error) {
	return s.timeSlotRepo.ListAll()
}

//
// ---------------- Generate ----------------
//

// GenerateTimeSlotsForWeek tạo time slots cho tất cả bác sĩ trong tuần tới (thứ 2 đến thứ 6)
func (s *TimeSlotService) GenerateTimeSlotsForWeek() error {
	doctors, err := s.doctorRepo.List()
	if err != nil {
		return err
	}

	nextWeekStart := s.getNextWeekStart(time.Now())

	// Chỉ tạo time slots cho thứ 2 đến thứ 6 (5 ngày làm việc)
	for i := 0; i < 5; i++ {
		date := nextWeekStart.AddDate(0, 0, i)
		if err := s.generateTimeSlotsForDate(doctors, date); err != nil {
			return err
		}
	}
	log.Printf("✅ Generated time slots for all doctors for week starting %v (Monday to Friday)", nextWeekStart)
	return nil
}

// GenerateTimeSlotsForMonth tạo time slots cho tất cả bác sĩ trong tháng tới (chỉ các ngày làm việc thứ 2 đến thứ 6)
func (s *TimeSlotService) GenerateTimeSlotsForMonth() error {
	doctors, err := s.doctorRepo.List()
	if err != nil {
		return err
	}

	nextMonthStart := s.getNextMonthStart(time.Now())
	nextMonthEnd := nextMonthStart.AddDate(0, 1, 0).Add(-time.Second) // Cuối tháng

	log.Printf("🗓️ Generating time slots for month: %v to %v", 
		nextMonthStart.Format("2006-01-02"), nextMonthEnd.Format("2006-01-02"))

	// Duyệt qua tất cả các ngày trong tháng
	currentDate := nextMonthStart
	workdaysCount := 0
	
	for currentDate.Before(nextMonthEnd) || currentDate.Equal(nextMonthEnd) {
		// Chỉ tạo time slots cho các ngày làm việc (thứ 2 đến thứ 6)
		if currentDate.Weekday() >= time.Monday && currentDate.Weekday() <= time.Friday {
			if err := s.generateTimeSlotsForDate(doctors, currentDate); err != nil {
				log.Printf("⚠️ Error generating slots for date %v: %v", currentDate.Format("2006-01-02"), err)
			} else {
				workdaysCount++
			}
		}
		currentDate = currentDate.AddDate(0, 0, 1) // Chuyển sang ngày tiếp theo
	}

	log.Printf("✅ Generated time slots for all doctors for month %v/%v (%d workdays)", 
		nextMonthStart.Month(), nextMonthStart.Year(), workdaysCount)
	return nil
}

// GenerateTimeSlotsForDoctor tạo slot tuần cho 1 bác sĩ (thứ 2 đến thứ 6)
func (s *TimeSlotService) GenerateTimeSlotsForDoctor(doctorID string) error {
	doctor, err := s.doctorRepo.FindByID(doctorID)
	if err != nil {
		return err
	}

	nextWeekStart := s.getNextWeekStart(time.Now())

	// Chỉ tạo time slots cho thứ 2 đến thứ 6 (5 ngày làm việc)
	for i := 0; i < 5; i++ {
		date := nextWeekStart.AddDate(0, 0, i)
		if err := s.generateTimeSlotsForDate([]Doctor{*doctor}, date); err != nil {
			return err
		}
	}
	return nil
}

//
// ---------------- Helper ----------------
//

func (s *TimeSlotService) generateTimeSlotsForDate(doctors []Doctor, date time.Time) error {
	for _, doctor := range doctors {
		startOfDay := time.Date(date.Year(), date.Month(), date.Day(), 0, 0, 0, 0, date.Location())
		endOfDay := startOfDay.Add(24 * time.Hour)

		// Xóa slot cũ
		if err := s.timeSlotRepo.DeleteByDoctorIDAndDateRange(doctor.DoctorID, startOfDay, endOfDay); err != nil {
			continue
		}

		// Tạo slot mới
		timeSlots := s.createTimeSlotsForDoctor(doctor.DoctorID, date)
		if len(timeSlots) > 0 {
			if err := s.timeSlotRepo.CreateBatch(timeSlots); err != nil {
				continue
			}
		}
	}
	return nil
}

func (s *TimeSlotService) createTimeSlotsForDoctor(doctorID string, date time.Time) []appointment.TimeSlot {
	var timeSlots []appointment.TimeSlot

	shifts := []struct {
		startHour, startMinute int
		endHour, endMinute     int
	}{
		{8, 30, 9, 0},
		{9, 0, 10, 0},
		{10, 0, 11, 0},
		{11, 0, 12, 0},
		{13, 30, 14, 0},
		{14, 0, 15, 0},
		{15, 0, 16, 0},
		{16, 0, 17, 0},
		{18, 0, 19, 0},
		{19, 0, 20, 0},
		{20, 0, 21, 0},
	}

	for _, shift := range shifts {
		startTime := time.Date(date.Year(), date.Month(), date.Day(), shift.startHour, shift.startMinute, 0, 0, date.Location())
		endTime := time.Date(date.Year(), date.Month(), date.Day(), shift.endHour, shift.endMinute, 0, 0, date.Location())

		timeSlot := appointment.TimeSlot{
			SlotID:    generateSlotID(),
			DoctorID:  doctorID,
			StartTime: startTime,
			EndTime:   endTime,
			Capacity:  1,
			CreatedAt: time.Now(),
			UpdatedAt: time.Now(),
		}
		timeSlots = append(timeSlots, timeSlot)
	}
	return timeSlots
}

func (s *TimeSlotService) getNextWeekStart(now time.Time) time.Time {
	daysUntilMonday := (8 - int(now.Weekday())) % 7
	if daysUntilMonday == 0 {
		daysUntilMonday = 7
	}
	nextMonday := now.AddDate(0, 0, daysUntilMonday)
	return time.Date(nextMonday.Year(), nextMonday.Month(), nextMonday.Day(), 0, 0, 0, 0, nextMonday.Location())
}

// getNextMonthStart trả về ngày đầu tiên của tháng tiếp theo
func (s *TimeSlotService) getNextMonthStart(now time.Time) time.Time {
	// Lấy ngày 1 của tháng tiếp theo
	nextMonth := now.AddDate(0, 1, 0)
	return time.Date(nextMonth.Year(), nextMonth.Month(), 1, 0, 0, 0, 0, nextMonth.Location())
}

func (s *TimeSlotService) GetByDoctorAndMonth(doctorID string, date time.Time) ([]appointment.TimeSlot, error) {
	return s.timeSlotRepo.FindByDoctorAndMonth(doctorID, date)
}

func (s *TimeSlotService) GetByDoctorAndDate(doctorID string, date time.Time) ([]appointment.TimeSlot, error) {
	return s.timeSlotRepo.FindByDoctorAndDate(doctorID, date)
}

func (s *TimeSlotService) GetByDoctorAndDateRange(doctorID string, startDate, endDate time.Time) ([]appointment.TimeSlot, error) {
	return s.timeSlotRepo.FindByDoctorAndDateRange(doctorID, startDate, endDate)
}
	

func generateSlotID() string {
	return uuid.NewString()
}


func (s *TimeSlotService) CreateBatch(doctorID string, inputs []TimeSlotInput) ([]appointment.TimeSlot, error) {
	if doctorID == "" {
		return nil, errors.New("doctorID is required")
	}
	if len(inputs) == 0 {
		return nil, errors.New("no time slots provided")
	}

	var slots []appointment.TimeSlot
	now := time.Now()

	for _, in := range inputs {
		if in.Capacity <= 0 {
			return nil, errors.New("capacity must be greater than 0")
		}
		if !in.StartTime.Before(in.EndTime) {
			return nil, errors.New("startTime must be before endTime")
		}

		// check overlap
		count, err := s.timeSlotRepo.CountOverlapping(doctorID, in.StartTime, in.EndTime)
		if err != nil {
			return nil, fmt.Errorf("failed to check overlapping: %w", err)
		}
		if count > 0 {
			return nil, fmt.Errorf("doctor already has a timeslot in %v - %v", in.StartTime, in.EndTime)
		}

		slots = append(slots, appointment.TimeSlot{
			SlotID:    generateSlotID(),
			DoctorID:  doctorID,
			StartTime: in.StartTime,
			EndTime:   in.EndTime,
			Capacity:  in.Capacity,
			CreatedAt: now,
			UpdatedAt: now,
		})
	}

	if err := s.timeSlotRepo.CreateBatch(slots); err != nil {
		return nil, err
	}

	return slots, nil
}
// Định nghĩa khung giờ các ca
var shiftDefinitions = map[string][]struct {
	startHour, startMinute int
	endHour, endMinute     int
}{
	"morning": {
		{8, 30, 9, 0},   // 08:30 - 09:00
		{9, 0, 10, 0},   // 09:00 - 10:00
		{10, 0, 11, 0},  // 10:00 - 11:00
		{11, 0, 12, 0},  // 11:00 - 12:00
	},
	"afternoon": {
		{13, 30, 14, 0}, // 13:30 - 14:00
		{14, 0, 15, 0},  // 14:00 - 15:00
		{15, 0, 16, 0},  // 15:00 - 16:00
		{16, 0, 17, 0},  // 16:00 - 17:00
	},
	"evening": {
		{18, 0, 19, 0},  // 18:00 - 19:00
		{19, 0, 20, 0},  // 19:00 - 20:00
		{20, 0, 21, 0},  // 20:00 - 21:00
	},
}

func (s *TimeSlotService) CreateMultiShiftSlots(req CreateMultiShiftSlotsRequest) ([]appointment.TimeSlot, error) {
    var allSlots []appointment.TimeSlot
  	loc, _ := time.LoadLocation("Asia/Ho_Chi_Minh")
    for _, sel := range req.Shifts {
        date, err := time.Parse("2006-01-02", sel.Date)
        if err != nil {
            return nil, fmt.Errorf("invalid date: %s", sel.Date)
        }

        for _, shift := range sel.Slots {
            defs, ok := shiftDefinitions[shift]
            if !ok {
                return nil, fmt.Errorf("invalid shift: %s", shift)
            }

            for _, def := range defs {
                start := time.Date(date.Year(), date.Month(), date.Day(),
					def.startHour, def.startMinute, 0, 0, loc)
				end := time.Date(date.Year(), date.Month(), date.Day(),
     				def.endHour, def.endMinute, 0, 0, loc)             

                slot := appointment.TimeSlot{
                    SlotID:    generateSlotID(),
                    DoctorID:  req.DoctorID,
                    StartTime: start,
                    EndTime:   end,
                    Capacity:  1,
                    CreatedAt: time.Now(),
                    UpdatedAt: time.Now(),
                }
                allSlots = append(allSlots, slot)
            }
        }
    }

    if err := s.timeSlotRepo.CreateBatch(allSlots); err != nil {
        return nil, err
    }

    return allSlots, nil
}

func (s *TimeSlotService) ImportDoctorDayOff(filePath string) error {
	f, err := excelize.OpenFile(filePath)
	if err != nil {
		return fmt.Errorf("failed to open excel: %w", err)
	}

	sheetName := f.GetSheetName(0)
	rows, err := f.GetRows(sheetName)
	if err != nil {
		return fmt.Errorf("failed to read rows: %w", err)
	}

	for i, row := range rows {
		if i == 0 || len(row) < 4 {
			continue
		}

		doctorID := strings.TrimSpace(row[0])
		dayOffList := strings.TrimSpace(row[2])
		shiftList := strings.TrimSpace(row[3])
		if doctorID == "" || dayOffList == "" {
			continue
		}

		days := strings.Split(dayOffList, ";")
		shifts := strings.Split(shiftList, ";")

		for idx, d := range days {
			d = strings.TrimSpace(d)
			if d == "" {
				continue
			}
			day, err := time.Parse("02/01/2006", d)
			if err != nil {
				return fmt.Errorf("lỗi parse ngày '%s' tại dòng %d: %v", d, i+1, err)
			}

			var shift string
			if idx < len(shifts) {
				shift = strings.ToLower(strings.TrimSpace(shifts[idx]))
			} else {
				shift = "all"
			}

			var ranges [][2]time.Time
			if shift == "all" {
				start := time.Date(day.Year(), day.Month(), day.Day(), 0, 0, 0, 0, time.Local)
				ranges = append(ranges, [2]time.Time{start, start.Add(24 * time.Hour)})
			} else if defs, ok := shiftDefinitions[shift]; ok {
				for _, def := range defs {
					start := time.Date(day.Year(), day.Month(), day.Day(), def.startHour, def.startMinute, 0, 0, time.Local)
					end := time.Date(day.Year(), day.Month(), day.Day(), def.endHour, def.endMinute, 0, 0, time.Local)
					ranges = append(ranges, [2]time.Time{start, end})
				}
			} else {
				return fmt.Errorf("buổi '%s' không hợp lệ tại dòng %d", shift, i+1)
			}

			for _, rg := range ranges {
				start, end := rg[0], rg[1]
				slots, err := s.timeSlotRepo.FindByDoctorIDAndDateRange(doctorID, start, end)
				if err != nil {
					return fmt.Errorf("lỗi lấy slot bác sĩ %s: %v", doctorID, err)
				}

				for _, sl := range slots {
					if sl.AppointmentID != nil && *sl.AppointmentID != "" {
						fmt.Printf("⚠️ Slot %s có appointment, đang xử lý thay thế...\n", sl.SlotID)

						// 🔍 tìm bác sĩ cùng chuyên khoa + hospital
						altDoctor, err := s.doctorRepo.FindBestReplacementDoctor(doctorID, sl.StartTime, sl.EndTime)
						if err != nil {
							return fmt.Errorf("lỗi tìm bác sĩ thay thế cho %s: %v", doctorID, err)
						}
						if altDoctor == nil {
							return fmt.Errorf("🚫 Không tìm thấy bác sĩ cùng chuyên khoa & hospital có slot trống [%s - %s]",
								sl.StartTime.Format("15:04"), sl.EndTime.Format("15:04"))
						}

						existingSlots, err := s.timeSlotRepo.FindByDoctorIDAndDateRange(
							altDoctor.DoctorID,
							time.Date(sl.StartTime.Year(), sl.StartTime.Month(), sl.StartTime.Day(), 0, 0, 0, 0, time.Local),
							time.Date(sl.StartTime.Year(), sl.StartTime.Month(), sl.StartTime.Day(), 23, 59, 59, 0, time.Local),
						)
						if err != nil {
							return fmt.Errorf("lỗi tìm slot bác sĩ mới %s: %v", altDoctor.DoctorID, err)
						}

						var selectedSlot *appointment.TimeSlot
						for _, es := range existingSlots {
							if (es.AppointmentID == nil || *es.AppointmentID == "") &&
								es.StartTime.Equal(sl.StartTime) && es.EndTime.Equal(sl.EndTime) {
								selectedSlot = &es
								break
							}
						}

						if selectedSlot == nil {
							for _, es := range existingSlots {
								if es.AppointmentID == nil || *es.AppointmentID == "" {
									if isSameShift(sl.StartTime, es.StartTime) {
										selectedSlot = &es
										break
									}
								}
							}
						}

						if selectedSlot == nil {
							return fmt.Errorf("🚫 Không có slot phù hợp trong ngày cho bác sĩ %s (%s)",
								altDoctor.DoctorID, altDoctor.FullName)
						}

						appt, err := s.appointmentRepo.GetByID(*sl.AppointmentID)
						if err != nil {
							return fmt.Errorf("lỗi lấy appointment %s: %v", *sl.AppointmentID, err)
						}

						appt.DoctorID = altDoctor.DoctorID
						if err := s.appointmentRepo.Update(appt); err != nil {
							return fmt.Errorf("lỗi cập nhật appointment %s sang bác sĩ %s: %v",
								*sl.AppointmentID, altDoctor.DoctorID, err)
						}

						selectedSlot.AppointmentID = sl.AppointmentID
						if err := s.timeSlotRepo.Update(selectedSlot); err != nil {
							return fmt.Errorf("lỗi cập nhật slot thay thế %s: %v", selectedSlot.SlotID, err)
						}

						if err := s.timeSlotRepo.Delete(sl.SlotID); err != nil {
							fmt.Printf("⚠️ Lỗi xoá slot cũ %s: %v\n", sl.SlotID, err)
						}

						fmt.Printf("🔄 Appointment %s chuyển sang bác sĩ %s (%s) slot %s [%s→%s]\n",
							*sl.AppointmentID, altDoctor.DoctorID, altDoctor.FullName,
							selectedSlot.SlotID,
							selectedSlot.StartTime.Format("15:04"), selectedSlot.EndTime.Format("15:04"),
						)
						continue
					}

					// Nếu không có appointment → xoá slot
					if err := s.timeSlotRepo.Delete(sl.SlotID); err != nil {
						fmt.Printf("❌ Lỗi xoá slot %s: %v\n", sl.SlotID, err)
					} else {
						fmt.Printf("✅ Đã xoá slot %s (doctor %s)\n", sl.SlotID, doctorID)
					}
				}
			}
		}
	}

	return nil
}


func isSameShift(t1, t2 time.Time) bool {
	h1, h2 := t1.Hour(), t2.Hour()
	switch {
	case h1 < 12 && h2 < 12:
		return true // sáng
	case h1 >= 12 && h1 < 17 && h2 >= 12 && h2 < 17:
		return true // chiều
	case h1 >= 17 && h2 >= 17:
		return true // tối
	default:
		return false
	}
}
