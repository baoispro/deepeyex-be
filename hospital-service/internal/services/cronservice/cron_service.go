package cronservice

import (
	"hospital-service/internal/services/appointmentservice"
	"log"
	"time"

	"github.com/robfig/cron/v3"
)

type CronService struct {
	timeSlotService *appointmentservice.TimeSlotService
	cron            *cron.Cron
}

func NewCronService(timeSlotService *appointmentservice.TimeSlotService) *CronService {
	// Tạo cron với timezone Việt Nam
	loc, _ := time.LoadLocation("Asia/Ho_Chi_Minh")
	c := cron.New(cron.WithLocation(loc))

	return &CronService{
		timeSlotService: timeSlotService,
		cron:            c,
	}
}

// Start khởi động cron service
func (s *CronService) Start() error {
	// Thêm job chạy vào thứ 7 hàng tuần lúc 23:00 để tạo lịch cho tuần tiếp theo
	// Cron expression: "0 23 * * 6" (phút giờ ngày tháng thứ) - 6 = thứ 7
	_, err := s.cron.AddFunc("0 23 * * 6", s.generateWeeklyTimeSlots)
	if err != nil {
		log.Printf("Error adding cron job: %v", err)
		return err
	}

	s.cron.Start()
	log.Println("Cron service started successfully - Will run every Saturday at 23:00")
	return nil
}

// Stop dừng cron service
func (s *CronService) Stop() {
	s.cron.Stop()
	log.Println("Cron service stopped")
}

// generateWeeklyTimeSlots job chính để generate time slots hàng tuần
func (s *CronService) generateWeeklyTimeSlots() {
	log.Println("Starting weekly time slots generation...")
	
	startTime := time.Now()
	err := s.timeSlotService.GenerateTimeSlotsForWeek()
	duration := time.Since(startTime)
	
	if err != nil {
		log.Printf("Error generating weekly time slots: %v (took %v)", err, duration)
	} else {
		log.Printf("Successfully generated weekly time slots (took %v)", duration)
	}
}

// testJob job test để kiểm tra cron hoạt động
func (s *CronService) testJob() {
	log.Println("Test cron job executed at", time.Now().Format("2006-01-02 15:04:05"))
}

// immediateTestJob job test chạy mỗi 30 giây để debug
func (s *CronService) immediateTestJob() {
	log.Println("IMMEDIATE TEST: Cron is working! Time:", time.Now().Format("2006-01-02 15:04:05"))
}

// GetCronEntries trả về danh sách các cron entries đang chạy
func (s *CronService) GetCronEntries() []cron.Entry {
	return s.cron.Entries()
}
