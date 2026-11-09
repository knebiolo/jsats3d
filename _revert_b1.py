import pathlib, sys
sys.path.insert(0, 'scripts')
import clock_jump_pre
result = clock_jump_pre.revert_tdoa_supplement(
    pathlib.Path(r'C:\Dev\jsat3d\jsats3d\output\11082026_whatif_B1\jsats3d_2025_B1.db'),
    pathlib.Path(r'C:\Dev\jsat3d\jsats3d\output\10292026_Test\jsats3d_2025_earlyfish.db')
)
print('Result:', result)
assert result['removed_stage2_events'] == 1653, 'Expected 1653, got %d' % result['removed_stage2_events']
print('removed_stage2_events = 1653 (correct)')
