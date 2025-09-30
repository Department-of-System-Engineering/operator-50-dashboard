from typing import List
from typing import Optional
from sqlalchemy import ForeignKey
from sqlalchemy import String
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
class Base(DeclarativeBase):
    pass
class VideoData(Base):
    __tablename__ = "video_data"
    id: Mapped[int] = mapped_column(primary_key=True)
    video_length: Mapped[float] = mapped_column()
    left_side_score: Mapped[float] = mapped_column()
    right_side_score: Mapped[float] = mapped_column()

    # Kapcsolatok visszafelé
    risk_maps: Mapped[List["RiskMap"]] = relationship(back_populates="video")
    critical_movements: Mapped[List["CriticalMovements"]] = relationship(back_populates="video")

    def __repr__(self) -> str:
        return f"VideoData(id={self.id!r}, video_length={self.video_length!r}, left_side_score={self.left_side_score!r}, right_side_score={self.right_side_score!r})"
class OperatorData(Base):
    __tablename__ = "operator_data"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column()
    date: Mapped[str] = mapped_column()
    assessor: Mapped[str] = mapped_column()
    task : Mapped[str] = mapped_column()
    working_duration: Mapped[float] = mapped_column()
    assessment_type: Mapped[str] = mapped_column()
    workstation: Mapped[str] = mapped_column() 
    def __repr__(self) -> str:
       return f"OperatorData(id={self.id!r}, name={self.name!r}, date={self.date!r}, assessor={self.assessor!r}, task={self.task!r}, working_duration={self.working_duration!r}, assessment_type={self.assessment_type!r}, workstation={self.workstation!r})"
    
class RiskMap(Base):
    __tablename__ = "risk_map"
    id: Mapped[int] = mapped_column(primary_key=True)
    video_id: Mapped[int] = mapped_column(ForeignKey("video_data.id"))
    video: Mapped[VideoData] = relationship(back_populates="risk_maps")
    neck: Mapped[float] = mapped_column()
    trunk: Mapped[float] = mapped_column()
    leg: Mapped[float] = mapped_column()
    upper_arm: Mapped[float] = mapped_column()
    lower_arm: Mapped[float] = mapped_column()
    wrist: Mapped[float] = mapped_column()
    def __repr__(self) -> str:
        return f"RiskMap(id={self.id!r}, video_id={self.video_id!r})"
    
class CriticalMovements(Base):
    __tablename__ = "critical_movements"
    id: Mapped[int] = mapped_column(primary_key=True)
    video_id: Mapped[int] = mapped_column(ForeignKey("video_data.id"))
    video: Mapped[VideoData] = relationship(back_populates="critical_movements")
    left_score: Mapped[float] = mapped_column()
    right_score: Mapped[float] = mapped_column()
    msd_risk_level: Mapped[float] = mapped_column()
    action_required: Mapped[str] = mapped_column()
    def __repr__(self) -> str:
        return f"CriticalMovements(id={self.id!r}, video_id={self.video_id!r})"