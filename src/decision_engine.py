"""
LBW Decision Engine based on MCC Law 36
"""

class LBWDecisionEngine:
    """Implements MCC Law 36 for LBW decisions"""
    
    def __init__(self, stump_data):
        """
        Args:
            stump_data: Dict with stump positions
        """
        self.stump_middle_x = stump_data['Stump_Middle_X']
        self.stump_width = stump_data['Stump_Width']
        self.stump_left_boundary = stump_data['Stump_Left_Boundary']
        self.stump_right_boundary = stump_data['Stump_Right_Boundary']
    
    def make_decision(self, pitch_x, impact_x, hits_stumps, shot_offered=True):
        """
        Make LBW decision based on MCC Law 36
        
        Args:
            pitch_x: X coordinate of ball pitch
            impact_x: X coordinate of ball impact
            hits_stumps: Whether trajectory hits stumps
            shot_offered: Whether batsman offered a shot
            
        Returns:
            Dict with decision details
        """
        # 1. Pitch in line
        pitched_in_line = pitch_x <= self.stump_right_boundary
        pitch_status = "LEGAL" if pitched_in_line else "ILLEGAL-Outside Leg"
        
        # 2. Impact in line
        if impact_x > self.stump_right_boundary:
            impact_in_line = False
            impact_status = "ILLEGAL-Outside Leg"
        elif shot_offered and impact_x < self.stump_left_boundary:
            impact_in_line = False
            impact_status = "ILLEGAL-Outside Off"
        else:
            impact_in_line = True
            impact_status = "LEGAL"
        
        # 3. Hitting stumps
        wicket_status = "HITTING" if hits_stumps else "MISSING"
        
        # Final decision
        is_out = pitched_in_line and impact_in_line and hits_stumps
        verdict = 'OUT' if is_out else 'NOT OUT'
        
        # Build response
        response = {
            'Pitch_Status': pitch_status,
            'Impact_Status': impact_status,
            'Wicket_Status': wicket_status,
            'Verdict': verdict,
            'Is_Out': 1 if is_out else 0,
            'Details': {
                'pitched_in_line': pitched_in_line,
                'impact_in_line': impact_in_line,
                'hits_stumps': hits_stumps
            }
        }
        
        # Add reasons if not out
        if not is_out:
            reasons = []
            if not pitched_in_line:
                reasons.append("Pitched outside leg")
            if not impact_in_line:
                reasons.append("Impact not in line")
            if not hits_stumps:
                reasons.append("Ball missing stumps")
            response['Reasons'] = reasons
        
        return response
    
    def print_decision(self, decision):
        """Pretty print the decision"""
        print(f"\n{'='*60}")
        print(f"🏏 LBW DECISION (MCC Law 36)")
        print(f"{'='*60}")
        print(f"   Pitch: {decision['Pitch_Status']}")
        print(f"   Impact: {decision['Impact_Status']}")
        print(f"   Wicket: {decision['Wicket_Status']}")
        print(f"\n📋 VERDICT: {'🟥 OUT' if decision['Is_Out'] else '🟩 NOT OUT'}")
        
        if not decision['Is_Out'] and 'Reasons' in decision:
            print(f"   Reason: {'; '.join(decision['Reasons'])}")
        print(f"{'='*60}")